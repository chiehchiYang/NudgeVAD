"""Generate doScenes submission.csv from a trained VAD-LLaVA-h4f12 model.

For every (instruction, scene) pair in the v1.0-test split (~493), find the
anchor sample (5th frame), build the VAD inference batch, override the
LLaVA prompt with the doScenes instruction, run forward, convert the
predicted offsets to ego frame X-forward/Y-left, and append a row.

Final CSV header:
    sample_token,x1,y1,x2,y2,...,x12,y12

Coordinates are in the local ego frame at anchor t=0 (X forward, Y left)
in metres, per the doScenes spec.

Usage:
    python tools/test_doscenes.py \\
        --config projects/configs/VAD/VAD_tiny_e2e_llava_h4f12.py \\
        --ckpt   output_doscenes_h4f12_v2/epoch_30.pth \\
        --output vis_doscenes/submission_v2.csv

Produces both with-language and (optionally) baseline CSVs for ΔADE
reporting on the official leaderboard.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
sys.path.insert(0, 'third_party/doScenes')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector

from nuscenes.nuscenes import NuScenes
from dataloader import DoScenesNuScenesDataset, load_paths

from eval_doscenes_local import _get_meta  # noqa: E402


# ---------- coordinate helpers ----------


def cumsum_pos(offsets: np.ndarray) -> np.ndarray:
    return np.cumsum(offsets, axis=-2)


def lidar_to_ego(xy: np.ndarray, lidar2ego_R: np.ndarray = None) -> np.ndarray:
    """LiDAR frame -> doScenes ego frame (X forward, Y left).

    If `lidar2ego_R` (the 3x3 rotation block of the per-sample lidar2ego
    matrix) is provided, applies that rotation precisely (only the 2D X/Y
    part of the trajectory is touched). Otherwise falls back to the
    canonical -90° rotation around z (the nominal nuScenes LIDAR_TOP
    convention; sub-degree imprecise).
    """
    if lidar2ego_R is not None:
        # Apply 2D rotation from per-sample lidar2ego matrix
        # lidar2ego: [x_ego, y_ego, ...] = R @ [x_lidar, y_lidar, 0, ...]
        # For trajectory deltas / relative positions we drop translation
        R2 = np.asarray(lidar2ego_R, dtype=np.float64)[:2, :2]
        out = xy @ R2.T   # (T, 2) @ (2, 2)
        return out.astype(xy.dtype)
    # Fallback: canonical -90° rotation
    out = np.zeros_like(xy)
    out[..., 0] = xy[..., 1]
    out[..., 1] = -xy[..., 0]
    return out


def _unwrap_to_tensor(obj):
    for _ in range(20):
        if hasattr(obj, 'data') and hasattr(obj, 'cpu_only'):
            obj = obj.data
        elif isinstance(obj, list) and obj:
            obj = obj[0]
        else:
            break
    return obj


# ---------- inference ----------


def _reset_prev_frame_info(model):
    inner = model.module if hasattr(model, 'module') else model
    inner.prev_frame_info = {
        'prev_bev': None,
        'scene_token': None,
        'prev_pos': 0,
        'prev_angle': 0,
    }


def _save_can_bus(meta) -> np.ndarray:
    if 'can_bus' in meta:
        return np.asarray(meta['can_bus'], dtype=np.float64).copy()
    return None


def _restore_can_bus(meta, can_bus_orig):
    if can_bus_orig is not None and 'can_bus' in meta:
        meta['can_bus'] = can_bus_orig.copy()


def predict_offsets(model, batch, instruction: str) -> np.ndarray:
    """Run a single forward with the given instruction text and return the
    predicted 12-step offsets in LiDAR frame.  Resets prev_frame_info and
    restores can_bus so repeated calls on the same batch are deterministic.
    """
    meta = _get_meta(batch)
    can_bus_orig = _save_can_bus(meta)
    _restore_can_bus(meta, can_bus_orig)
    meta['ego_instruction'] = instruction or ''
    meta['ego_instruction_present'] = bool(instruction)
    _reset_prev_frame_info(model)

    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **batch)
    res = out[0] if isinstance(out, list) else out
    pts = res.get('pts_bbox', {}) if isinstance(res, dict) else {}

    wp = pts.get('llava_waypoints')
    if isinstance(wp, torch.Tensor) and wp.shape == (12, 2):
        return wp.cpu().numpy()

    # fallback: VAD ego planner (12, 2) selected by current ego_fut_cmd
    efp = pts.get('ego_fut_preds')
    if isinstance(efp, torch.Tensor):
        efp = efp.cpu().numpy()
    cmd = _unwrap_to_tensor(batch['ego_fut_cmd'])
    cmd_arr = cmd.cpu().numpy() if hasattr(cmd, 'cpu') else np.asarray(cmd)
    cmd_idx = int(np.argmax(np.squeeze(cmd_arr)))
    return efp[cmd_idx]


# ---------- main ----------


def build_test_dataset(cfg, test_pkl):
    """Configure cfg.data.test to point at the h4f12 test pkl, build dataset."""
    cfg = Config(cfg.copy())  # deep copy so we don't mutate caller's cfg
    test_cfg = cfg.data.test.copy()
    test_cfg.pop('samples_per_gpu', None)
    test_cfg['ann_file'] = test_pkl
    return build_dataset(test_cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',  default='projects/configs/VAD/VAD_tiny_e2e_llava_h4f12.py')
    p.add_argument('--ckpt',    required=True)
    p.add_argument('--gpu',     type=int, default=0)
    p.add_argument('--test-pkl', default='data/nuscenes/vad_nuscenes_h4f12_infos_temporal_test.pkl')
    p.add_argument('--paths-txt', default='third_party/doScenes/paths.txt',
                   help='doScenes paths.txt with NUSCENES_ROOT / DOSCENES_ANNOTATIONS')
    p.add_argument('--output',  required=True,
                   help='submission CSV path (with-language pass)')
    p.add_argument('--baseline-output', default='',
                   help='if set, also produce a baseline (no-language) CSV')
    p.add_argument('--limit',   type=int, default=-1,
                   help='process at most N pairs (debugging)')
    p.add_argument('--progress-every', type=int, default=25)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # 1. Enumerate doScenes test pairs (493 expected)
    nusc_root, doscenes_ann = load_paths(args.paths_txt)
    nusc = NuScenes(version='v1.0-test', dataroot=nusc_root, verbose=False)
    ds_doscenes = DoScenesNuScenesDataset(
        nusc=nusc,
        annotations=doscenes_ann,
        camera_channels=('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'),
        include_blank_instructions=False,
    )
    print(f'[test] doScenes test pairs: {len(ds_doscenes)}')

    pairs: List[Dict[str, Any]] = []
    for i in range(len(ds_doscenes)):
        item = ds_doscenes[i]
        pairs.append({
            'scene_token':   item['scene_token'],
            'scene_name':    item['scene_name'],
            'instruction':   item['instruction'],
            'instruction_type': item['instruction_type'],
            'anchor_sample_token': item['anchor_sample_token'],
        })

    # 2. Build VAD test dataset (using val pipeline, single-frame)
    cfg = Config.fromfile(args.config)
    plugin = cfg.get('plugin_dir', 'projects/mmdet3d_plugin/').replace('/', '.').rstrip('.')
    importlib.import_module(plugin)
    # Only enable llava when the model class actually accepts it (VADLLaVA).
    # Pure VAD baseline configs (type='VAD') don't have this kwarg.
    if cfg.model.get('type', '') in ('VADLLaVA', 'VAD_LLaVA', 'VADLlava'):
        cfg.model.llava_enabled = True

    dataset = build_test_dataset(cfg, args.test_pkl)
    print(f'[test] VAD test dataset {type(dataset).__name__}, len={len(dataset)}')

    # build sample_token → dataset_idx map
    tok_to_idx = {info['token']: i for i, info in enumerate(dataset.data_infos)}
    print(f'[test] sample_token map size: {len(tok_to_idx)}')

    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4,
                              dist=False, shuffle=False)

    # 3. Build model
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # Only VADLLaVA needs lazy LLaVA init; pure VAD baseline doesn't have it.
    if hasattr(model, '_lazy_init_llava'):
        print('[test] forcing _lazy_init_llava() so LoRA layers load correctly')
        if not model._lazy_init_llava():
            raise RuntimeError(getattr(model, '_llava_runtime_error', 'llava init failed'))
    print(f'[test] load ckpt {args.ckpt}')
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)['state_dict']
    n_lora_ckpt = sum(1 for k in sd if 'lora_' in k.lower())
    n_lora_model = sum(1 for n, _ in model.named_parameters() if 'lora_' in n.lower())
    print(f'[test] LoRA keys ckpt={n_lora_ckpt} model={n_lora_model}')
    load_checkpoint(model, args.ckpt, map_location='cpu', strict=False)
    # Force ALL submodules to cuda (some PEFT/lazy modules stay on cpu
    # despite model.cuda()). Bypass MMDataParallel's strict device check.
    model = model.cuda(args.gpu)
    for m in model.modules():
        try:
            m.to(f'cuda:{args.gpu}')
        except Exception:
            pass

    class _SingleGPUWrapper(MMDataParallel):
        def __init__(self, module, device_ids):
            # Skip MMDataParallel.__init__ to bypass strict device check
            torch.nn.Module.__init__(self)
            self.module = module
            self.device_ids = device_ids
            self.dim = 0
            self.src_device_obj = torch.device(f'cuda:{device_ids[0]}')
            self.output_device = device_ids[0]
        def forward(self, *inputs, **kwargs):
            inputs_s, kwargs_s = self.scatter(inputs, kwargs, self.device_ids)
            return self.module(*inputs_s[0], **kwargs_s[0])
    model = _SingleGPUWrapper(model, device_ids=[args.gpu])
    model.eval()

    # 4. For each pair, fetch the anchor batch once; run with-language and
    #    optionally baseline.  We cache by anchor_sample_token because some
    #    scenes have multiple instructions sharing the same anchor.
    rows_lang: List[List[str]] = []
    rows_base: List[List[str]] = []
    skipped_no_anchor = 0
    started = time.time()

    # Get every distinct anchor we need; build a fresh batch when we hit it.
    # Use the dataloader iter to walk samples in pkl order.
    needed_tokens = {p['anchor_sample_token'] for p in pairs}
    print(f'[test] distinct anchor sample_tokens needed: {len(needed_tokens)}')

    # group pairs by anchor for grouping inferences
    pairs_by_anchor: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        pairs_by_anchor.setdefault(p['anchor_sample_token'], []).append(p)

    n_done = 0
    n_total = len(pairs) if args.limit < 0 else min(args.limit, len(pairs))

    for idx, batch in enumerate(loader):
        meta = _get_meta(batch)
        token = meta.get('sample_idx', '')
        if token not in pairs_by_anchor:
            continue

        # We have a needed anchor. Save the immutable can_bus once.
        can_bus_orig = _save_can_bus(meta)

        # Per-sample lidar2ego rotation (use actual cs_record rotation,
        # not canonical -90°, for sub-degree precision in coordinate frame)
        lidar2ego_mat = meta.get('lidar2ego', None)
        if lidar2ego_mat is not None:
            lidar2ego_R = np.asarray(lidar2ego_mat)[:3, :3]
        else:
            lidar2ego_R = None
        for p in pairs_by_anchor[token]:
            inst = p['instruction']
            scene_tok = p['scene_token']    # per doScenes spec, write scene_token (not sample/anchor token)
            # with-language pass
            _restore_can_bus(meta, can_bus_orig)
            offsets_lang = predict_offsets(model, batch, instruction=inst)
            pos_lang_lidar = cumsum_pos(offsets_lang)
            pos_lang_ego = lidar_to_ego(pos_lang_lidar, lidar2ego_R)
            row_lang = [scene_tok, inst]
            for x, y in pos_lang_ego:
                row_lang.append(f'{float(x):.6f}')
                row_lang.append(f'{float(y):.6f}')
            rows_lang.append(row_lang)

            if args.baseline_output:
                _restore_can_bus(meta, can_bus_orig)
                offsets_base = predict_offsets(model, batch, instruction='')
                pos_base_ego = lidar_to_ego(cumsum_pos(offsets_base), lidar2ego_R)
                row_base = [scene_tok, inst]
                for x, y in pos_base_ego:
                    row_base.append(f'{float(x):.6f}')
                    row_base.append(f'{float(y):.6f}')
                rows_base.append(row_base)

            n_done += 1
            if args.progress_every > 0 and n_done % args.progress_every == 0:
                elapsed = time.time() - started
                rate = n_done / elapsed if elapsed > 0 else 0.0
                print(f'  [{n_done}/{n_total}] elapsed={elapsed:.0f}s '
                      f'rate={rate:.2f}/s  inst="{inst[:60]}"')

            if args.limit > 0 and n_done >= args.limit:
                break

        if args.limit > 0 and n_done >= args.limit:
            break

    skipped_no_anchor = len(pairs) - n_done
    print(f'\n[test] processed {n_done} pairs, skipped {skipped_no_anchor} '
          f'(anchor not in test pkl)')

    # 5. Write CSVs
    # Header per official doScenes spec:
    #   sample_token,instruction,x1,y1,x2,y2,...,x12,y12
    # "sample_token" column carries the scene_token value (per the spec).
    header = ['sample_token', 'instruction']
    for i in range(1, 13):
        header.extend([f'x{i}', f'y{i}'])

    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_lang)
    print(f'[test] wrote with-language submission -> {args.output}  ({len(rows_lang)} rows)')

    if args.baseline_output:
        with open(args.baseline_output, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows_base)
        print(f'[test] wrote baseline submission -> {args.baseline_output}  ({len(rows_base)} rows)')


if __name__ == '__main__':
    main()
