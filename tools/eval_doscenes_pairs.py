"""Per-(scene, instruction) pair val eval.

For each val anchor (frame_idx==4) that has a doScenes annotation:
  - With-language: forward model once per instruction text → records N pairs
  - No-language:   forward model once (blanked instruction) → replicates same
                   prediction to all N instructions of that scene

Output npz contains per-pair arrays of shape (N_pairs,) plus pred_offsets
(N_pairs, 12, 2) so a downstream summary can aggregate metrics.

Handles VAD's stateful prev_frame_info: snapshots before the first anchor
forward, restores between per-instruction iterations so each forward sees
identical temporal context.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import importlib
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector


sys.path.insert(0, '.')
from tools.eval_doscenes_local import (
    _get_meta, _cumsum_to_positions, _per_step_l2,
    _neutralize_cmd_inplace, _should_apply_stop_rule)


def _load_scene_to_instructions(ann_dir, scene_token_to_name_json):
    with open(scene_token_to_name_json) as f:
        tok2name = json.load(f)
    name2tok = {v: k for k, v in tok2name.items()}
    out = {}
    for path in sorted(glob.glob(os.path.join(ann_dir, '*.csv'))):
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                raw = (row.get('Scene Number') or '').strip()
                inst = (row.get('Instruction') or '').strip()
                itype = (row.get('Instruction Type') or '').strip()
                if not raw or not inst:
                    continue
                try:
                    num = int(float(raw))
                except ValueError:
                    continue
                name = f'scene-{num:04d}'
                tok = name2tok.get(name)
                if tok:
                    out.setdefault(tok, []).append((inst, itype))
    return out


def _override_instruction(meta, inst, itype):
    meta['ego_instruction'] = inst
    meta['ego_instruction_type'] = itype
    meta['ego_instruction_present'] = bool(inst)
    meta['has_static_reference'] = 's' in (itype or '').lower()
    meta['has_dynamic_reference'] = 'd' in (itype or '').lower()


def _blank_instruction(meta):
    meta['ego_instruction'] = ''
    meta['ego_instruction_type'] = ''
    meta['ego_instruction_present'] = False
    meta['has_static_reference'] = False
    meta['has_dynamic_reference'] = False


def _extract_pred_offsets(out, batch):
    res = out[0] if isinstance(out, list) else out
    if not isinstance(res, dict) or 'pts_bbox' not in res:
        return None
    pts = res['pts_bbox']
    pred = None
    if 'llava_waypoints' in pts and pts['llava_waypoints'] is not None:
        wp = pts['llava_waypoints']
        if isinstance(wp, torch.Tensor) and wp.shape == (12, 2):
            pred = wp.cpu().numpy()
    if pred is None:
        efp = pts['ego_fut_preds']
        if isinstance(efp, torch.Tensor):
            efp = efp.cpu().numpy()
        cmd = batch['ego_fut_cmd']
        for _ in range(20):
            if hasattr(cmd, 'data') and hasattr(cmd, 'cpu_only'):
                cmd = cmd.data
            elif isinstance(cmd, list) and cmd:
                cmd = cmd[0]
            else:
                break
        cmd_arr = cmd.cpu().numpy() if hasattr(cmd, 'cpu') else np.asarray(cmd)
        cmd_arr = np.squeeze(cmd_arr)
        cmd_idx = int(np.argmax(cmd_arr))
        pred = efp[cmd_idx]
    return pred


def _extract_gt_offsets(batch):
    gt = batch['ego_fut_trajs']
    for _ in range(20):
        if hasattr(gt, 'data') and hasattr(gt, 'cpu_only'):
            gt = gt.data
        elif isinstance(gt, list) and gt:
            gt = gt[0]
        else:
            break
    arr = gt.cpu().numpy() if hasattr(gt, 'cpu') else np.asarray(gt)
    return np.squeeze(arr)


def _extract_history_speed(batch):
    his = batch.get('ego_his_trajs')
    for _ in range(20):
        if hasattr(his, 'data') and hasattr(his, 'cpu_only'):
            his = his.data
        elif isinstance(his, list) and his:
            his = his[0]
        else:
            break
    if his is None:
        return 0.0
    arr = his.cpu().numpy() if hasattr(his, 'cpu') else np.asarray(his)
    arr = np.squeeze(arr).reshape(-1, 2)
    if not arr.size:
        return 0.0
    return float(np.linalg.norm(arr, axis=-1).max() / 0.5)


def _snapshot_prev_frame_info(model_module):
    """Return a deep-ish copy of prev_frame_info (preserving prev_bev tensor identity)."""
    return {
        'scene_token': model_module.prev_frame_info['scene_token'],
        'prev_bev': model_module.prev_frame_info['prev_bev'],
        'prev_pos': copy.deepcopy(model_module.prev_frame_info.get('prev_pos')),
        'prev_angle': copy.deepcopy(model_module.prev_frame_info.get('prev_angle')),
    }


def _restore_prev_frame_info(model_module, snap):
    model_module.prev_frame_info['scene_token'] = snap['scene_token']
    model_module.prev_frame_info['prev_bev'] = snap['prev_bev']
    model_module.prev_frame_info['prev_pos'] = copy.deepcopy(snap['prev_pos'])
    model_module.prev_frame_info['prev_angle'] = copy.deepcopy(snap['prev_angle'])


def _snapshot_can_bus(meta):
    cb = meta.get('can_bus')
    return np.asarray(cb).copy() if cb is not None else None


def _restore_can_bus(meta, snap):
    if snap is None:
        return
    cb = meta.get('can_bus')
    if cb is not None:
        cb[...] = snap


def run_pairs_eval(cfg_path, ckpt_path, gpu_id, use_language,
                   dump_path, apply_stop_rule=False, anchor_window=4,
                   keep_cmd_no_lang=False):
    cfg = Config.fromfile(cfg_path)
    plugin = cfg.get('plugin_dir', 'projects/mmdet3d_plugin/').replace('/', '.').rstrip('.')
    importlib.import_module(plugin)
    is_vad_only = (cfg.model.type == 'VAD')
    if not is_vad_only:
        cfg.model.llava_enabled = True

    d_cfg = cfg.data.val.copy()
    d_cfg.pop('samples_per_gpu', None)
    d_cfg['doscenes_anchor_only'] = False
    d_cfg['doscenes_anchor_frames'] = list(range(anchor_window + 1))

    ann_dir = None
    scene_json = None
    def _walk(pipe):
        nonlocal ann_dir, scene_json
        for t in pipe:
            if isinstance(t, dict):
                if t.get('type') == 'LoadDoScenesInstruction':
                    ann_dir = ann_dir or t.get('ann_dir')
                    scene_json = scene_json or t.get('scene_token_to_name_json')
                if t.get('type') == 'MultiScaleFlipAug3D':
                    _walk(t.get('transforms', []))
    _walk(d_cfg['pipeline'])
    # Fallback for VAD-only configs that don't include LoadDoScenesInstruction
    # (e.g. Stage 1 baseline). Use the canonical project paths.
    if not ann_dir:
        ann_dir = 'third_party/doScenes/Annotations'
    if not scene_json:
        scene_json = 'data/nuscenes/scene_token_to_name.json'

    scene_to_inst = _load_scene_to_instructions(ann_dir, scene_json)
    print(f'[pairs] doScenes: {sum(len(v) for v in scene_to_inst.values())} pairs '
          f'across {len(scene_to_inst)} scenes')

    dataset = build_dataset(d_cfg)
    print(f'[pairs] dataset len={len(dataset)}')
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4,
                              dist=False, shuffle=False)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if not is_vad_only:
        if not model._lazy_init_llava():
            raise RuntimeError('LLaVA init failed')
    print(f'[pairs] loading ckpt: {ckpt_path}')
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    model = model.cuda(gpu_id)
    target_dev = torch.device(f'cuda:{gpu_id}')
    for mod in model.modules():
        try:
            mod.to(target_dev)
        except Exception:
            pass

    class _Wrap(MMDataParallel):
        def forward(self, *inputs, **kwargs):
            inputs_s, kwargs_s = self.scatter(inputs, kwargs, self.device_ids)
            if not inputs_s and not kwargs_s:
                inputs_s = ((),); kwargs_s = ({},)
            return self.module(*inputs_s[0], **kwargs_s[0])
    model = _Wrap(model, device_ids=[gpu_id])
    model.eval()

    records: List[Dict] = []
    n_stop_rule = 0
    started = time.time()

    for idx, batch in enumerate(loader):
        meta = _get_meta(batch)
        frame_idx = int(meta.get('frame_idx', -1))
        scene_tok = meta.get('scene_token', '')

        # For non-anchor frames: forward once to accumulate prev_bev, no recording.
        if frame_idx != 4:
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **batch)
            continue

        inst_list = scene_to_inst.get(scene_tok, [])
        if not inst_list:
            # Anchor but no doScenes instruction; still forward to keep state
            # in case downstream samples expect anchor in scene.
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **batch)
            continue

        gt_offsets = _extract_gt_offsets(batch)
        if gt_offsets.shape != (12, 2):
            continue
        gt_pos = _cumsum_to_positions(gt_offsets)
        his_speed = _extract_history_speed(batch)
        sample_tok = str(meta.get('sample_idx', ''))

        # Snapshot temporal state + can_bus before the first anchor forward.
        snap_pfi = _snapshot_prev_frame_info(model.module)
        snap_cb = _snapshot_can_bus(meta)

        if not use_language:
            _blank_instruction(meta)
            if not keep_cmd_no_lang:
                # Default: blank cmd to [0,0,1] (Go Straight) — strictest
                # no-leakage baseline.
                _neutralize_cmd_inplace(batch)
            # else: keep whatever the pipeline produced. In nocmd configs the
            # pipeline runs ForceCmdNeutral(mode='random'), so cmd is a random
            # one-hot matching training distribution → own-baseline ΔADE
            # measures the pure language contribution.
            with torch.no_grad():
                out = model(return_loss=False, rescale=True, **batch)
            pred_offsets = _extract_pred_offsets(out, batch)
            if pred_offsets is None:
                continue
            pred_pos = _cumsum_to_positions(pred_offsets)
            l2 = _per_step_l2(pred_pos, gt_pos)
            # Replicate to every instruction of this scene
            for inst, itype in inst_list:
                records.append({
                    'scene_token': scene_tok, 'sample_token': sample_tok,
                    'instruction': inst, 'instruction_type': itype,
                    'pred_offsets': pred_offsets.copy(),
                    'gt_offsets': gt_offsets.copy(),
                    'step_err': l2.copy(),
                    'history_speed': his_speed,
                })
        else:
            for inst, itype in inst_list:
                # Restore state so every instruction sees the same prev_bev / can_bus
                _restore_prev_frame_info(model.module, snap_pfi)
                _restore_can_bus(meta, snap_cb)
                _override_instruction(meta, inst, itype)
                with torch.no_grad():
                    out = model(return_loss=False, rescale=True, **batch)
                pred_offsets = _extract_pred_offsets(out, batch)
                if pred_offsets is None:
                    continue
                if apply_stop_rule and _should_apply_stop_rule(inst, his_speed):
                    pred_offsets = np.zeros_like(pred_offsets)
                    n_stop_rule += 1
                pred_pos = _cumsum_to_positions(pred_offsets)
                l2 = _per_step_l2(pred_pos, gt_pos)
                records.append({
                    'scene_token': scene_tok, 'sample_token': sample_tok,
                    'instruction': inst, 'instruction_type': itype,
                    'pred_offsets': pred_offsets.copy(),
                    'gt_offsets': gt_offsets.copy(),
                    'step_err': l2.copy(),
                    'history_speed': his_speed,
                })

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - started
            print(f'  [{idx + 1} iters / {len(records)} pairs] elapsed={elapsed:.0f}s', flush=True)

    n = len(records)
    print(f'[pairs] collected {n} pair records ({n_stop_rule} stop-rule fires)')
    if n == 0:
        return
    arr_step = np.stack([r['step_err'] for r in records], axis=0)
    arr_pred = np.stack([r['pred_offsets'] for r in records], axis=0)
    arr_gt = np.stack([r['gt_offsets'] for r in records], axis=0)
    np.savez(
        dump_path,
        step_err=arr_step,
        pred_offsets=arr_pred,
        gt_offsets=arr_gt,
        scene_tokens=np.array([r['scene_token'] for r in records]),
        sample_tokens=np.array([r['sample_token'] for r in records]),
        instructions=np.array([r['instruction'] for r in records]),
        instruction_types=np.array([r['instruction_type'] for r in records]),
        history_speeds=np.array([r['history_speed'] for r in records]),
    )
    print(f'[pairs] dumped → {dump_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--with-language', action='store_true')
    p.add_argument('--no-language', action='store_true')
    p.add_argument('--dump-prefix', required=True)
    p.add_argument('--apply-stop-rule', action='store_true')
    p.add_argument('--anchor-window', type=int, default=4)
    p.add_argument('--keep-cmd-no-lang', action='store_true',
                   help='In the --no-language pass, only blank the instruction '
                        'string but KEEP the cmd produced by the val pipeline '
                        '(under nocmd configs that is a random one-hot from '
                        'ForceCmdNeutral). Aligns the baseline cmd distribution '
                        'with training and isolates pure language contribution '
                        'in the own-baseline ΔADE.')
    args = p.parse_args()

    if not (args.with_language or args.no_language):
        args.with_language = True
        args.no_language = True

    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    if args.with_language:
        run_pairs_eval(args.config, args.ckpt, args.gpu, use_language=True,
                       dump_path=f'{args.dump_prefix}_with_language.npz',
                       apply_stop_rule=args.apply_stop_rule,
                       anchor_window=args.anchor_window)
    if args.no_language:
        run_pairs_eval(args.config, args.ckpt, args.gpu, use_language=False,
                       dump_path=f'{args.dump_prefix}_baseline.npz',
                       apply_stop_rule=False,
                       anchor_window=args.anchor_window,
                       keep_cmd_no_lang=args.keep_cmd_no_lang)


if __name__ == '__main__':
    main()
