"""Generate the 150-row 26-col submission CSV for the doScenes test set.

INFERENCE-ONLY: does not read `gt_ego_fut_trajs` from the test pkl, does not
compute ADE/FDE. The script's sole job is to write predictions in the
mi3-lab format so they can be uploaded to the official leaderboard.

Format:
  sample_token, instruction, x1, y1, x2, y2, ..., x12, y12

Outputs:
  - `<out_dir>/test150_NudgeVAD_FiLMv4_rule_v5.csv`     (with-lang + rule v5)
  - `<out_dir>/test150_NudgeVAD_FiLMv4_no_language.csv` (baseline)

Both passes are run from the same checkpoint. The script enumerates the 150
v1.0-test anchors via the test pkl, builds 5-frame queues so VAD's stateful
`prev_bev` is correctly seeded, then for each anchor runs two forwards
(with-language using the first doScenes instruction; no-language with
random cmd matching training distribution).
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib
import os
import pickle
import sys
import time

import numpy as np
import torch
from pyquaternion import Quaternion

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector


# Reuse helpers from eval_doscenes_local for batch unwrapping etc.
sys.path.insert(0, '.')
from tools.eval_doscenes_local import (
    _get_meta, _cumsum_to_positions,
    _neutralize_cmd_inplace, _should_apply_stop_rule)


def lidar_to_ego(xy, R):
    R2 = np.asarray(R, dtype=np.float64)[:2, :2]
    return (xy @ R2.T).astype(np.float64)


def _extract_pred_offsets(out, batch):
    res = out[0] if isinstance(out, list) else out
    if not isinstance(res, dict) or 'pts_bbox' not in res:
        return None
    pts = res['pts_bbox']
    wp = pts.get('llava_waypoints')
    if isinstance(wp, torch.Tensor) and wp.shape == (12, 2):
        return wp.cpu().numpy()
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
    return efp[int(np.argmax(cmd_arr))]


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True,
                   help='Override config that points val pipeline at the test pkl.')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--test-pkl',
                   default='data/nuscenes/vad_nuscenes_h4f12_infos_temporal_test.pkl')
    p.add_argument('--nuscenes-root', default='data/nuscenes')
    p.add_argument('--doscenes-ann', default='third_party/doScenes/Annotations')
    p.add_argument('--out-dir', default='submissions')
    p.add_argument('--apply-stop-rule', action='store_true',
                   help='Apply rule v5 stop override to the with-language pass.')
    args = p.parse_args()

    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    # 1. doScenes first instruction per scene
    sys.path.insert(0, 'third_party/doScenes')
    from nuscenes.nuscenes import NuScenes
    from dataloader import DoScenesNuScenesDataset
    nusc = NuScenes(version='v1.0-test', dataroot=args.nuscenes_root, verbose=False)
    ds_ann = DoScenesNuScenesDataset(
        nusc=nusc, annotations=args.doscenes_ann,
        camera_channels=('CAM_FRONT',), include_blank_instructions=False)
    scene_first_inst = {}
    for i in range(len(ds_ann)):
        item = ds_ann[i]
        scene_first_inst.setdefault(item['scene_token'], item['instruction'])

    # 2. Test pkl: lidar2ego per anchor
    with open(args.test_pkl, 'rb') as f:
        d = pickle.load(f)
    infos_all = d['infos'] if isinstance(d, dict) else d
    tok2info = {info['token']: info for info in infos_all}

    # 3. Build model + dataloader (5-frame queue per scene via doscenes_anchor_frames=[0..4])
    cfg = Config.fromfile(args.config)
    plugin = cfg.get('plugin_dir', 'projects/mmdet3d_plugin/').replace('/', '.').rstrip('.')
    importlib.import_module(plugin)
    if cfg.model.type == 'VADLLaVA':
        cfg.model.llava_enabled = True

    d_cfg = cfg.data.val.copy()
    d_cfg.pop('samples_per_gpu', None)
    d_cfg['doscenes_anchor_only'] = False
    d_cfg['doscenes_anchor_frames'] = [0, 1, 2, 3, 4]
    dataset = build_dataset(d_cfg)
    print(f'[predict] dataset len={len(dataset)} (expected 750 = 5 frames × 150 scenes)')

    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4,
                              dist=False, shuffle=False)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if cfg.model.type == 'VADLLaVA':
        if not model._lazy_init_llava():
            raise RuntimeError('LLaVA init failed')
    print(f'[predict] loading ckpt: {args.ckpt}')
    load_checkpoint(model, args.ckpt, map_location='cpu', strict=False)
    model = model.cuda(args.gpu)
    target_dev = torch.device(f'cuda:{args.gpu}')
    for mod in model.modules():
        try:
            mod.to(target_dev)
        except Exception:
            pass

    class _Wrap(MMDataParallel):
        def forward(self, *inputs, **kwargs):
            inputs_s, kwargs_s = self.scatter(inputs, kwargs, self.device_ids)
            if not inputs_s and not kwargs_s:
                inputs_s = ((),)
                kwargs_s = ({},)
            return self.module(*inputs_s[0], **kwargs_s[0])
    model = _Wrap(model, device_ids=[args.gpu])
    model.eval()

    # 4. Walk all 750 samples to keep prev_bev correct; record only at frame_idx==4
    pred_lang = {}      # scene_token -> (pred_offsets [12, 2], anchor_sample_token, his_speed)
    pred_nolang = {}
    started = time.time()
    for idx, batch in enumerate(loader):
        meta = _get_meta(batch)
        scene_tok = meta.get('scene_token', '')
        frame_idx = int(meta.get('frame_idx', -1))
        atok = meta.get('sample_idx', '')

        if frame_idx != 4:
            # forward (no record) to accumulate prev_bev
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **batch)
            continue

        # Anchor: run two passes. Snapshot batch state between them.
        first_inst = scene_first_inst.get(scene_tok, '')

        # ---- WITH-LANGUAGE pass ----
        meta['ego_instruction'] = first_inst
        meta['ego_instruction_present'] = bool(first_inst)
        if 'ego_instruction_type' in meta:
            meta['ego_instruction_type'] = ''
        if 'has_static_reference' in meta:
            meta['has_static_reference'] = False
        if 'has_dynamic_reference' in meta:
            meta['has_dynamic_reference'] = False

        snap_pfi = {
            'scene_token': model.module.prev_frame_info['scene_token'],
            'prev_bev': model.module.prev_frame_info['prev_bev'],
            'prev_pos': copy.deepcopy(model.module.prev_frame_info.get('prev_pos')),
            'prev_angle': copy.deepcopy(model.module.prev_frame_info.get('prev_angle')),
        }
        cb = meta.get('can_bus')
        snap_cb = np.asarray(cb).copy() if cb is not None else None

        with torch.no_grad():
            out = model(return_loss=False, rescale=True, **batch)
        po_lang = _extract_pred_offsets(out, batch)
        his_speed = _extract_history_speed(batch)

        # Restore state for no-language pass
        model.module.prev_frame_info['scene_token'] = snap_pfi['scene_token']
        model.module.prev_frame_info['prev_bev'] = snap_pfi['prev_bev']
        model.module.prev_frame_info['prev_pos'] = copy.deepcopy(snap_pfi['prev_pos'])
        model.module.prev_frame_info['prev_angle'] = copy.deepcopy(snap_pfi['prev_angle'])
        if snap_cb is not None and cb is not None:
            cb[...] = snap_cb

        # ---- NO-LANGUAGE pass ----
        meta['ego_instruction'] = ''
        meta['ego_instruction_present'] = False
        # Keep cmd from pipeline (random one-hot from ForceCmdNeutral)
        with torch.no_grad():
            out = model(return_loss=False, rescale=True, **batch)
        po_nolang = _extract_pred_offsets(out, batch)

        pred_lang[scene_tok] = (po_lang, atok, his_speed, first_inst)
        pred_nolang[scene_tok] = (po_nolang, atok)

        if (len(pred_lang) % 25) == 0:
            print(f'  [{len(pred_lang)}/150] elapsed={time.time()-started:.0f}s', flush=True)

    print(f'[predict] {len(pred_lang)} anchors recorded')

    # 5. Write CSVs
    HEADER = ['sample_token', 'instruction'] + \
             [f'{c}{i}' for i in range(1, 13) for c in ('x', 'y')]
    os.makedirs(args.out_dir, exist_ok=True)

    def write_csv(path, sub_dict, apply_rule_v5):
        n_fire = 0
        fired = []
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for scene_tok in sorted(sub_dict.keys()):
                entry = sub_dict[scene_tok]
                if len(entry) == 4:
                    off, atok, his_speed, inst = entry
                else:
                    off, atok = entry
                    inst, his_speed = '', 0.0
                R = Quaternion(tok2info[atok]['lidar2ego_rotation']).rotation_matrix
                off_ego = lidar_to_ego(off, R)
                pos_ego = np.cumsum(off_ego, axis=0)
                if apply_rule_v5 and _should_apply_stop_rule(
                        inst, history_speed_mps=his_speed):
                    pos_ego = np.zeros_like(pos_ego)
                    n_fire += 1
                    fired.append(inst)
                row = [scene_tok, inst]
                for i in range(12):
                    row.extend([f'{pos_ego[i, 0]:.6f}', f'{pos_ego[i, 1]:.6f}'])
                w.writerow(row)
        return n_fire, fired

    p_lang = os.path.join(args.out_dir, 'test150_NudgeVAD_FiLMv4_rule_v5.csv')
    n_fire, fired = write_csv(p_lang, pred_lang, apply_rule_v5=args.apply_stop_rule)
    p_no = os.path.join(args.out_dir, 'test150_NudgeVAD_FiLMv4_no_language.csv')
    write_csv(p_no, pred_nolang, apply_rule_v5=False)

    print(f'\nwrote {p_lang}  (rule v5 fired {n_fire} times)')
    for x in fired:
        print(f'  - "{x}"')
    print(f'wrote {p_no}')


if __name__ == '__main__':
    main()
