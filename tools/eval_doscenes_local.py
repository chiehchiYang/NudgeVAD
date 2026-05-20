"""Local doScenes ADE / FDE / miss_rate evaluator (VAL ONLY).

⚠️  DO NOT POINT THIS SCRIPT AT THE TEST PKL.  It reads `gt_ego_fut_trajs`
from the dataset infos to compute ADE / FDE — running it on test data
would compute test-set metrics locally, which violates the doScenes "no
test GT" rule. Use `tools/predict_test150.py` for test inference (it is
inference-only and never reads `gt_ego_fut_trajs`).

Runs a trained VAD-LLaVA-h4f12 model on the val split filtered to samples
that have a doScenes instruction, then reports ADE@2s / 4s / 6s, FDE, and
miss_rate.  ADE is rotation-invariant, so no coordinate-frame conversion is
needed.

Two passes are supported:
  --language        : feed the doScenes instruction to LLaVA AND keep the
                      pipeline-derived ego_fut_cmd (under v6 this cmd is
                      itself derived from the instruction text, so it's
                      compliance-safe — no GT-future leak).
  --no-language     : blank out instruction string AND neutralize cmd to
                      [0, 0, 1] (Go Straight) — the baseline gets NO
                      instruction signal at all. This makes ΔADE measure
                      "language + cmd combined effect" under the
                      compliance constraint.

If both flags are passed, runs both back-to-back and prints ΔADE.

Map metrics (offroad / offyaw) are intentionally skipped — they require the
NuScenesMap pre-loaded per-scene, which doubles the eval startup time.  Add
later via --with-map-metrics if needed.
"""
from __future__ import annotations

import argparse
import importlib
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


# ----------------------------- helpers -----------------------------


def _get_meta(batch) -> dict:
    """Walk any nested DataContainer / list wrapper until we hit a meta dict.

    Observed shapes in practice:
        train pipeline (queue=3): list -> DC -> list -> list -> {0: meta_q0, 1: meta_q1, 2: meta_q2}
        val pipeline (MSFA3D)   : list -> DC -> list -> list -> meta_dict
    Some collate paths add an extra outer list wrapper, so we just keep
    unwrapping until we find a meta dict (string keys).
    """
    m = batch['img_metas']
    for _ in range(20):  # safety bound on depth
        if hasattr(m, 'data') and hasattr(m, 'cpu_only'):
            m = m.data
            continue
        if isinstance(m, list):
            if not m:
                return {}
            m = m[0]
            continue
        if isinstance(m, dict):
            if not m:
                return {}
            first_key = next(iter(m))
            if isinstance(first_key, int):
                # queue dict — pick the anchor (highest index = latest frame)
                m = m[max(m.keys())]
                continue
            return m  # real meta dict with string keys
        return {}
    return {}


def _cumsum_to_positions(offsets: np.ndarray) -> np.ndarray:
    """Convert per-step offsets (T, 2) → cumulative positions relative to the anchor (T, 2)."""
    return np.cumsum(offsets, axis=-2)


def _per_step_l2(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1)


def _bin_centers_from_thresholds(thresholds, top_pad: float = 2.5) -> np.ndarray:
    """Derive bin centres for an N-threshold → (N+1)-class speed head.

    The first class spans [0, t0]  → centre = t0 / 2
    Mid classes span [t_i, t_{i+1}] → centre = midpoint
    Last class spans [t_{N-1}, ∞)  → centre = t_{N-1} + top_pad
    """
    t = np.asarray(thresholds, dtype=np.float64)
    centres = np.empty(len(t) + 1, dtype=np.float64)
    centres[0] = t[0] / 2
    for i in range(len(t) - 1):
        centres[i + 1] = (t[i] + t[i + 1]) / 2
    centres[-1] = t[-1] + top_pad
    return centres


def _apply_speed_correction(offsets: np.ndarray,
                            speed_logits: np.ndarray,
                            bin_centers: np.ndarray,
                            dt: float = 0.5) -> np.ndarray:
    """TransFuser++-style post-hoc speed correction.

    Scales each step's offset magnitude to match the speed bin's centre,
    keeping direction. Stops (cls 0) collapse the offset toward zero.

    Args:
        offsets: (T, 2)  per-step xy offsets in ego frame
        speed_logits: (T, 7)  per-step speed-class logits
        dt: step interval (s)

    Returns:
        (T, 2) corrected offsets.
    """
    T = offsets.shape[0]
    pred_class = np.argmax(speed_logits, axis=-1)             # (T,)
    target_speed = bin_centers[pred_class]                     # (T,)
    target_dist = target_speed * dt                            # (T,) m
    cur_dist = np.linalg.norm(offsets, axis=-1)                # (T,)
    out = offsets.copy()
    for t in range(T):
        c = float(cur_dist[t])
        if c < 1e-3:
            # Direction unknown; if speed says we should be moving,
            # we have no direction info → leave as-is (zero).
            continue
        scale = float(target_dist[t]) / c
        out[t] = offsets[t] * scale
    return out


# ----------------------------- rule v4 stop override -----------------------------

import re as _re

_HARD_STOP_RE = _re.compile(
    r'\b(stop|stops|stopping|halt|halts|halting|brake|brakes|braking|'
    r'yield|yields|yielding|wait|waits|waiting)\b', _re.IGNORECASE)
# Noun phrases that mention "stop" but do NOT instruct the ego to stop
_NOUN_STOP_RE = _re.compile(
    r'\b(stop\s+sign|stop\s+light|stop\s+line|bus\s+stop|stop\s+bar)\b',
    _re.IGNORECASE)
# If any of these motion verbs is present, the instruction is NOT a hard stop
# (e.g. "Go straight and stop", "Turn right and stop at the sign").
_OTHER_ACTION_RE = _re.compile(
    r'\b(go|goes|going|turn|turns|turning|drive|drives|driving|move|moves|moving|'
    r'merge|merges|merging|follow|follows|following|take|takes|taking|'
    r'change|changes|changing|switch|switches|switching|continue|continues|continuing|'
    r'proceed|proceeds|proceeding|pass|passes|passing|overtake|overtakes|overtaking|'
    r'accelerate|accelerates|accelerating|speed|speeds|speeding|'
    r'head|heads|heading|keep|keeps|keeping|reverse|reverses|reversing)\b',
    _re.IGNORECASE)


def _should_apply_stop_rule(instruction: str,
                            history_speed_mps: float = 0.0,
                            history_speed_max: float = 2.0) -> bool:
    """Rule v5: classify instruction as a HARD STOP command **and** verify the
    ego is already nearly stationary.

    Returns True iff ALL of:
      - instruction contains a HARD_STOP keyword (stop / halt / brake / yield / wait)
      - AND does NOT contain a noun-stop phrase (stop sign / bus stop / ...)
      - AND does NOT contain any OTHER_ACTION verb (go / turn / drive / ...)
      - AND word count <= 12 (long compound sentences are filtered out)
      - AND history_speed_mps <= history_speed_max (default 2 m/s ≈ 7 km/h):
        "stop" with a fast-moving ego usually means "eventually stop at the
        next intersection / behind the car", NOT "halt now". Gating by
        history speed removes those false positives.
    """
    if history_speed_mps > history_speed_max:
        return False
    if not isinstance(instruction, str):
        return False
    s = instruction.strip()
    if not s:
        return False
    if len(s.split()) > 12:
        return False
    if _NOUN_STOP_RE.search(s):
        return False
    if _OTHER_ACTION_RE.search(s):
        return False
    return bool(_HARD_STOP_RE.search(s))


def _neutralize_cmd_inplace(batch) -> int:
    """Overwrite batch['ego_fut_cmd'] tensor(s) in-place with the
    "default forward / straight" one-hot (LAST class = 1, others = 0).

    This is the doScenes "no leakage" baseline: when the language pass is
    blanked out, the model must also lose its direction hint via the cmd
    channel. Using [..., -1] = 1 keeps this generic across cmd schemes:
      * 3-class (v6): [..., 2] = Straight
      * 6-class (v8): [..., 5] = forward
    """
    n_overwritten = 0
    stack = [batch.get('ego_fut_cmd')]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if hasattr(node, 'data') and hasattr(node, 'cpu_only'):
            stack.append(node.data)
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, torch.Tensor):
            node.zero_()
            node[..., -1] = 1.0
            n_overwritten += 1
    return n_overwritten


def _summarise(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) if len(v) else float('nan') for k, v in metric_lists.items()}


# ----------------------------- main eval loop -----------------------------


def run_eval(cfg_path: str,
             ckpt_path: str,
             gpu_id: int,
             use_language: bool,
             max_samples: int = -1,
             only_doscenes: bool = True,
             anchor_only: bool = False,
             no_anchor: bool = False,
             use_gt_cmd: bool = False,
             progress_every: int = 100,
             dump_path: str = '',
             use_speed_correction: bool = False,
             keep_cmd_no_lang: bool = False,
             deterministic_instruction: bool = False,
             anchor_window: int = 0,
             apply_stop_rule: bool = False) -> Dict[str, float]:
    cfg = Config.fromfile(cfg_path)
    plugin = cfg.get('plugin_dir', 'projects/mmdet3d_plugin/').replace('/', '.').rstrip('.')
    importlib.import_module(plugin)

    # Force LLaVA on so we have a stable inference path; even when
    # --no-language we still go through the LLaVA branch but the model is
    # given an empty instruction (matching the doScenes ΔADE protocol).
    # VAD-only configs (model.type='VAD') don't need this — leave their
    # cfg untouched so build_detector picks the plain VAD class.
    is_vad_only = (cfg.model.type == 'VAD')
    if not is_vad_only:
        cfg.model.llava_enabled = True

    # Speed-bin centres derived from cfg so post-hoc correction always
    # matches whatever scheme the head was trained with.
    speed_bins_cfg = cfg.model.pts_bbox_head.get(
        'speed_bins', (0.5, 2.0, 5.0, 10.0, 15.0, 20.0))
    speed_bin_centers_arr = _bin_centers_from_thresholds(speed_bins_cfg)
    if use_speed_correction:
        print(f'[eval] speed bins from cfg: {tuple(speed_bins_cfg)}')
        print(f'[eval] derived bin centres: {speed_bin_centers_arr.round(2).tolist()}')

    # Build val dataset.  cfg.data.val passes through samples_per_gpu=1 which
    # the dataset class doesn't accept, so pop it.
    d_cfg = cfg.data.val.copy()
    d_cfg.pop('samples_per_gpu', None)
    if no_anchor:
        d_cfg['doscenes_anchor_only'] = False
        print('[eval] --no-anchor: dataset will expose ALL val samples (not just anchors)')

    if anchor_window > 0:
        # Build prev-frame context for each anchor so VAD's stateful prev_bev
        # (in forward_test) correctly accumulates from frame 0 up to frame 4.
        # Need contiguous frame_idx ∈ [0..4] = anchor_window+1 frames per scene
        # (5 with anchor_window=4 = queue_length+anchor).
        d_cfg['doscenes_anchor_only'] = False
        d_cfg['doscenes_anchor_frames'] = list(range(anchor_window + 1))
        print(f'[eval] --anchor-window={anchor_window}: dataset filtered to '
              f'frame_idx ∈ {d_cfg["doscenes_anchor_frames"]} so prev_bev is correct.')

    if deterministic_instruction:
        # Monkey-patch LoadDoScenesInstruction.__call__: seed numpy.random with
        # a stable hash of scene_token before the random pick so every eval
        # run gives the same instruction for the same sample. Keeps mode unchanged.
        from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import (
            LoadDoScenesInstruction)
        import hashlib, numpy as _np
        _orig_call = LoadDoScenesInstruction.__call__

        def _det_call(self, results):
            scene_token = results.get('scene_token') or ''
            seed = int(hashlib.md5(str(scene_token).encode()).hexdigest()[:8], 16)
            _state = _np.random.get_state()
            _np.random.seed(seed)
            try:
                return _orig_call(self, results)
            finally:
                _np.random.set_state(_state)

        LoadDoScenesInstruction.__call__ = _det_call
        print('[eval] --deterministic-instruction: monkey-patched '
              'LoadDoScenesInstruction.__call__ to seed numpy.random by '
              'md5(scene_token) before the random pick')

    if use_gt_cmd:
        # Strip cmd-overriding transforms from the pipeline so the cmd stays
        # at whatever the dataset placed there (= info['gt_ego_fut_cmd']).
        def _is_cmd_transform(t):
            if not isinstance(t, dict):
                return False
            if t.get('type') == 'LoadLaneletCmd':
                return True
            if t.get('type') == 'LoadDoScenesInstruction' and t.get('override_cmd_from_text'):
                return True
            return False
        original_n = len(d_cfg['pipeline'])
        d_cfg['pipeline'] = [t for t in d_cfg['pipeline'] if not _is_cmd_transform(t)]
        # Also walk MultiScaleFlipAug3D's nested transforms (defensive)
        for t in d_cfg['pipeline']:
            if isinstance(t, dict) and t.get('type') == 'MultiScaleFlipAug3D':
                inner = t.get('transforms') or []
                t['transforms'] = [tt for tt in inner if not _is_cmd_transform(tt)]
        print(f'[eval] --use-gt-cmd: stripped cmd-override transforms '
              f'({original_n} → {len(d_cfg["pipeline"])} top-level); '
              f'cmd will be GT-future-derived from pkl.')
    dataset = build_dataset(d_cfg)
    print(f'[eval] dataset: {type(dataset).__name__}, len={len(dataset)}')

    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4,
                              dist=False, shuffle=False)

    # Build model.  LLaVA/LoRA/adapter/plan_head are LAZILY constructed on the
    # first forward call.  We need them present BEFORE load_checkpoint, else
    # the 128 LoRA keys + planning_adapter keys end up as "unexpected" and the
    # trained weights are silently dropped.
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if not is_vad_only:
        print('[eval] forcing _lazy_init_llava() so LoRA layers exist before ckpt load')
        if not model._lazy_init_llava():
            raise RuntimeError(f'LLaVA init failed: {getattr(model, "_llava_runtime_error", None)}')
    print(f'[eval] loading ckpt: {ckpt_path}')
    ret = load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    if not is_vad_only:
        # Sanity: confirm trained LoRA weights actually loaded
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
        n_lora_in_ckpt = sum(1 for k in sd if 'lora_' in k.lower())
        n_lora_in_model = sum(1 for n, _ in model.named_parameters() if 'lora_' in n.lower())
        print(f'[eval] LoRA keys: ckpt={n_lora_in_ckpt}, model={n_lora_in_model}')
        assert n_lora_in_ckpt > 0 and n_lora_in_model == n_lora_in_ckpt, \
            f'LoRA load mismatch: ckpt={n_lora_in_ckpt}, model={n_lora_in_model}'
    # Force ALL parameters/buffers onto cuda for single-GPU eval.
    model = model.cuda(gpu_id)
    target_dev = torch.device(f'cuda:{gpu_id}')
    for mod in model.modules():
        try:
            mod.to(target_dev)
        except Exception:
            pass

    # Bypass MMDataParallel's strict pre-check (it iterates ALL params each
    # forward and any leftover CPU tensor — e.g. a non-leaf PEFT internal —
    # raises). For single-GPU eval we just want scatter behaviour.
    class _SingleGPUWrapper(MMDataParallel):
        def forward(self, *inputs, **kwargs):
            inputs_s, kwargs_s = self.scatter(inputs, kwargs, self.device_ids)
            if not inputs_s and not kwargs_s:
                inputs_s = ((),)
                kwargs_s = ({},)
            return self.module(*inputs_s[0], **kwargs_s[0])
    model = _SingleGPUWrapper(model, device_ids=[gpu_id])
    model.eval()

    ade_2s, ade_4s, ade_6s, fde, miss_rate = [], [], [], [], []
    n_stop_rule_fired = 0
    if apply_stop_rule and not use_language:
        print('[eval] --apply-stop-rule has no effect in the no-language pass '
              '(rule reads the instruction text); ignoring.')
    step_errs = []        # per-sample (T,) trajectory L2 error
    speed_logits_arr = []  # per-sample (T, K) speed CE logits
    gt_speeds_arr = []    # per-sample (T,) GT speed magnitude (m/s)
    pred_offsets_arr = [] # per-sample (T, 2) predicted xy offsets
    gt_offsets_arr   = [] # per-sample (T, 2) GT xy offsets
    sample_tokens    = [] # per-sample str  nuScenes sample token
    frame_idxs       = [] # per-sample int  frame_idx
    skipped_no_inst = 0
    skipped_non_anchor = 0
    skipped_invalid_fut = 0
    started = time.time()

    debug_every = max(1, progress_every)
    for idx, batch in enumerate(loader):
        if max_samples > 0 and len(ade_6s) >= max_samples:
            break

        meta = _get_meta(batch)
        instruction_present = bool(meta.get('ego_instruction_present', False))
        if idx < 5 or idx % debug_every == 0:
            print(f'  iter idx={idx}: scene={meta.get("scene_token", "?")[:8]} '
                  f'present={instruction_present} '
                  f'evaluated_so_far={len(ade_6s)}', flush=True)
        if only_doscenes and not instruction_present:
            skipped_no_inst += 1
            continue
        # Match the official doScenes test protocol: only the 5th keyframe of
        # each scene (`frame_idx == 4`) is the prediction anchor; mid-scene
        # samples are not part of the leaderboard task.
        if anchor_only and int(meta.get('frame_idx', -1)) != 4:
            skipped_non_anchor += 1
            continue

        # Optionally blank out the instruction for the baseline pass.
        # doScenes compliance: when language is dropped, also neutralize cmd
        # to [0,0,1] (Go Straight default) so the baseline gets NO instruction
        # signal at all — neither natural language nor 3-class direction hint.
        if not use_language:
            for k in ('ego_instruction', 'ego_instruction_type', 'has_static_reference', 'has_dynamic_reference'):
                if k in meta:
                    meta[k] = '' if isinstance(meta[k], str) else False
            meta['ego_instruction_present'] = False
            if not keep_cmd_no_lang:
                n_cmd_overwritten = _neutralize_cmd_inplace(batch)
                if idx < 3 and n_cmd_overwritten > 0:
                    print(f'    [no-lang] neutralized {n_cmd_overwritten} cmd tensor(s) → [0,0,1]', flush=True)
            else:
                if idx < 3:
                    print(f'    [no-lang] keep_cmd_no_lang=True → cmd from pipeline (lanelet-inferred) untouched', flush=True)

        with torch.no_grad():
            out = model(return_loss=False, rescale=True, **batch)
        res = out[0] if isinstance(out, list) else out
        if not isinstance(res, dict) or 'pts_bbox' not in res:
            if len(ade_6s) == 0:
                print(f'  DEBUG: idx={idx} skip — res type={type(res).__name__}, '
                      f'keys={list(res.keys()) if isinstance(res, dict) else "N/A"}', flush=True)
            continue

        pts = res['pts_bbox']
        if len(ade_6s) == 0:
            # one-time: dump what's actually in pts_bbox
            print(f'  DEBUG idx={idx} pts_bbox keys: {sorted(pts.keys())}', flush=True)
            for k in ('llava_waypoints', 'ego_fut_preds'):
                if k in pts:
                    v = pts[k]
                    if isinstance(v, torch.Tensor):
                        print(f'    {k}: Tensor{tuple(v.shape)} {v.dtype}', flush=True)
                    else:
                        print(f'    {k}: {type(v).__name__}={v!r}'[:200], flush=True)
        # Prefer LLaVA-conditioned waypoints when present and well-formed; fall
        # back to the VAD ego planner indexed by the GT command.
        pred_offsets = None
        if 'llava_waypoints' in pts and pts['llava_waypoints'] is not None:
            wp = pts['llava_waypoints']
            if isinstance(wp, torch.Tensor) and wp.shape == (12, 2):
                pred_offsets = wp.cpu().numpy()
        if pred_offsets is None:
            efp = pts['ego_fut_preds']  # [3, 12, 2] in LiDAR/ego frame, per-step offsets
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
            pred_offsets = efp[cmd_idx]

        # Optional: TransFuser++-style post-hoc speed correction.
        # Rescale each step's offset magnitude to the speed-class
        # bin-centre predicted by the speed head, keeping direction.
        if use_speed_correction and 'ego_speed_logits' in pts:
            sl = pts['ego_speed_logits']
            if isinstance(sl, torch.Tensor):
                sl = sl.cpu().numpy()
            # Pick the cmd-mode (matches how trajectory mode was picked)
            if sl.ndim == 3 and sl.shape[0] >= 1:
                # cmd_idx may not be defined if we took the LLaVA path; default 0
                if 'cmd_idx' not in dir():
                    cmd = batch.get('ego_fut_cmd')
                    for _ in range(20):
                        if hasattr(cmd, 'data') and hasattr(cmd, 'cpu_only'):
                            cmd = cmd.data
                        elif isinstance(cmd, list) and cmd:
                            cmd = cmd[0]
                        else:
                            break
                    arr = cmd.cpu().numpy() if hasattr(cmd, 'cpu') else np.asarray(cmd)
                    arr = np.squeeze(arr)
                    cmd_idx = int(np.argmax(arr))
                cmd_idx_speed = min(cmd_idx, sl.shape[0] - 1)
                speed_logits_t = sl[cmd_idx_speed]   # (T, num_bins)
                pred_offsets = _apply_speed_correction(
                    pred_offsets.astype(np.float64),
                    speed_logits_t,
                    bin_centers=speed_bin_centers_arr,
                    dt=0.5,
                ).astype(np.float32)
                if len(ade_6s) == 0:
                    print(f'  [speed-correction] applied; '
                          f'pred_class[t]={np.argmax(speed_logits_t, axis=-1).tolist()}',
                          flush=True)

        # Rule v5 post-processing: when the instruction is a hard STOP command
        # AND the ego is already nearly stationary (history speed <= 2 m/s),
        # zero out the predicted offsets so the trajectory holds at the anchor.
        # Only active in the with-language pass (uses the instruction text).
        if apply_stop_rule and use_language:
            instr_txt = meta.get('ego_instruction', '') or ''
            # Compute history speed from batch['ego_his_trajs'] (per-step offsets,
            # ego frame). Take max step-speed as the "current motion" estimate.
            his_speed = 0.0
            his = batch.get('ego_his_trajs')
            for _ in range(20):
                if hasattr(his, 'data') and hasattr(his, 'cpu_only'):
                    his = his.data
                elif isinstance(his, list) and his:
                    his = his[0]
                else:
                    break
            if his is not None:
                his_arr = his.cpu().numpy() if hasattr(his, 'cpu') else np.asarray(his)
                his_arr = np.squeeze(his_arr).reshape(-1, 2)
                if his_arr.size:
                    his_speed = float(np.linalg.norm(his_arr, axis=-1).max() / 0.5)
            if _should_apply_stop_rule(instr_txt, history_speed_mps=his_speed):
                pred_offsets = np.zeros_like(pred_offsets)
                n_stop_rule_fired += 1
                if n_stop_rule_fired <= 5:
                    print(f'  [stop-rule] fired @ idx={idx}: '
                          f'his_v={his_speed:.2f} m/s "{instr_txt[:80]}"', flush=True)

        # Ground-truth offsets — robustly walk DC/list nesting like _get_meta().
        gt = batch['ego_fut_trajs']
        for _ in range(20):
            if hasattr(gt, 'data') and hasattr(gt, 'cpu_only'):
                gt = gt.data
            elif isinstance(gt, list) and gt:
                gt = gt[0]
            else:
                break
        gt_arr = gt.cpu().numpy() if hasattr(gt, 'cpu') else np.asarray(gt)
        gt_offsets = np.squeeze(gt_arr)  # (12, 2)
        if len(ade_6s) == 0:
            print(f'  DEBUG idx={idx} gt raw type={type(batch["ego_fut_trajs"]).__name__}'
                  f', gt_arr.shape={getattr(gt_arr, "shape", "?")}'
                  f', gt_offsets.shape={getattr(gt_offsets, "shape", "?")}'
                  f', pred_offsets.shape={pred_offsets.shape}', flush=True)
        if gt_offsets.shape != (12, 2):
            skipped_invalid_fut += 1
            continue

        # Cumsum to positions; ADE in LiDAR frame is identical to ADE in ego
        # frame because L2 is rotation-invariant.
        pred_pos = _cumsum_to_positions(pred_offsets)
        gt_pos = _cumsum_to_positions(gt_offsets)

        l2 = _per_step_l2(pred_pos, gt_pos)
        step_errs.append(l2.astype(np.float64))
        # Predicted + GT offsets, sample identity (for downstream submissions / vis)
        pred_offsets_arr.append(pred_offsets.astype(np.float64))
        gt_offsets_arr.append(gt_offsets.astype(np.float64))
        sample_tokens.append(str(meta.get('sample_idx', '') or ''))
        frame_idxs.append(int(meta.get('frame_idx', -1)))
        # Speed-head dump: (T, K) logits + GT per-step speed magnitude.
        # GT speed = ||per-step offset|| / dt (dt=0.5 s for h4f12).
        gt_step_speed = np.linalg.norm(gt_offsets, axis=-1) / 0.5  # (T,)
        gt_speeds_arr.append(gt_step_speed.astype(np.float64))
        if 'ego_speed_logits' in pts and pts['ego_speed_logits'] is not None:
            sl = pts['ego_speed_logits']
            if isinstance(sl, torch.Tensor):
                sl = sl.cpu().numpy()
            # Pick cmd-mode (sl shape [M, T, K])
            cmd_idx_safe = locals().get('cmd_idx', 0)
            cmd_idx_safe = min(cmd_idx_safe, sl.shape[0] - 1)
            speed_logits_arr.append(sl[cmd_idx_safe].astype(np.float64))
        else:
            speed_logits_arr.append(np.zeros((12, 1), dtype=np.float64))
        ade_2s.append(float(np.mean(l2[:4])))
        ade_4s.append(float(np.mean(l2[:8])))
        ade_6s.append(float(np.mean(l2)))
        fde.append(float(l2[-1]))
        miss_rate.append(float(np.max(l2) >= 2.0))

        n = len(ade_6s)
        if progress_every > 0 and n % progress_every == 0:
            elapsed = time.time() - started
            rate = n / elapsed if elapsed > 0 else 0.0
            print(f'  [{n}] elapsed={elapsed:.0f}s  rate={rate:.2f} sample/s  '
                  f'running ADE@6s={np.mean(ade_6s):.3f}  FDE={np.mean(fde):.3f}')

    summary_input = {
        'ade_2s':     ade_2s,
        'ade_4s':     ade_4s,
        'ade_6s':     ade_6s,
        'fde':        fde,
        'miss_rate':  miss_rate,
    }
    if step_errs:
        step_arr_for_summary = np.stack(step_errs, axis=0)
        for t in range(12):
            summary_input[f'ade_pt{t + 1}'] = step_arr_for_summary[:, t]
    summary = _summarise(summary_input)
    summary['n_eval']       = float(len(ade_6s))
    summary['skipped_no_inst'] = float(skipped_no_inst)
    summary['skipped_non_anchor'] = float(skipped_non_anchor)
    summary['skipped_invalid'] = float(skipped_invalid_fut)
    summary['wall_seconds'] = float(time.time() - started)
    summary['stop_rule_fired'] = float(n_stop_rule_fired)
    if apply_stop_rule and use_language:
        print(f'[eval] rule-v4 stop override fired on {n_stop_rule_fired} '
              f'/ {len(ade_6s)} samples')

    # Optionally dump per-sample arrays so we can compare with-lang vs baseline
    # element-wise (catches identical-rounded-but-actually-different cases).
    if dump_path:
        step_arr = (np.stack(step_errs, axis=0) if step_errs
                    else np.zeros((0, 12), dtype=np.float64))
        speed_arr = (np.stack(speed_logits_arr, axis=0) if speed_logits_arr
                     else np.zeros((0, 12, 1), dtype=np.float64))
        gt_speed_arr = (np.stack(gt_speeds_arr, axis=0) if gt_speeds_arr
                        else np.zeros((0, 12), dtype=np.float64))
        np.savez(
            dump_path,
            ade_2s=np.asarray(ade_2s, dtype=np.float64),
            ade_4s=np.asarray(ade_4s, dtype=np.float64),
            ade_6s=np.asarray(ade_6s, dtype=np.float64),
            fde=np.asarray(fde, dtype=np.float64),
            miss_rate=np.asarray(miss_rate, dtype=np.float64),
            step_err=step_arr,                    # (n_samples, 12) per-step L2
            ade_12pt=step_arr.mean(axis=0),        # (12,) mean ADE at 0.5s..6.0s
            speed_logits=speed_arr,               # (n_samples, 12, K) per-step speed CE logits
            gt_speed=gt_speed_arr,                 # (n_samples, 12) GT speed (m/s)
            speed_bins=np.asarray(speed_bins_cfg, dtype=np.float64),
            speed_bin_centres=speed_bin_centers_arr,
            # Raw predictions for downstream submission / visualisation
            pred_offsets=(np.stack(pred_offsets_arr, axis=0)
                          if pred_offsets_arr else np.zeros((0, 12, 2))),
            gt_offsets=(np.stack(gt_offsets_arr, axis=0)
                        if gt_offsets_arr else np.zeros((0, 12, 2))),
            sample_tokens=np.asarray(sample_tokens),
            frame_idxs=np.asarray(frame_idxs, dtype=np.int64),
        )
        print(f'[eval] per-sample arrays saved -> {dump_path}')
    return summary


def _print_block(title: str, summary: Dict[str, float]) -> None:
    print('\n' + '=' * 64)
    print(title)
    print('=' * 64)
    for k in ('n_eval', 'skipped_no_inst', 'skipped_non_anchor', 'skipped_invalid',
              'ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate', 'wall_seconds'):
        if k in summary:
            # high precision so tiny ΔADE differences don't get rounded away
            print(f'  {k:>16s} = {summary[k]:.6f}')
    pt_keys = [f'ade_pt{i}' for i in range(1, 13)]
    if all(k in summary for k in pt_keys):
        print('  ADE@12pts (0.5s..6.0s):')
        vals = '  '.join(f'{summary[k]:.3f}' for k in pt_keys)
        print(f'    {vals}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='projects/configs/VAD/VAD_tiny_e2e_llava_h4f12.py')
    p.add_argument('--ckpt', required=True, help='trained checkpoint .pth')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-samples', type=int, default=-1,
                   help='limit number of samples evaluated (debugging)')
    p.add_argument('--with-language', action='store_true',
                   help='run the language pass (instruction + text-derived cmd)')
    p.add_argument('--no-language',   action='store_true',
                   help='run the baseline pass (blank instruction + cmd '
                        'neutralized to [0,0,1] Go Straight)')
    p.add_argument('--keep-cmd-no-lang', action='store_true',
                   help='in --no-language pass, only blank the instruction text '
                        'but KEEP cmd from pipeline (lanelet-inferred). This '
                        'isolates pure language contribution to ΔADE.')
    p.add_argument('--all-samples',   action='store_true',
                   help='evaluate every val sample, not just doScenes-annotated ones')
    p.add_argument('--anchor-only',   action='store_true',
                   help='match doScenes test protocol: only frame_idx==4 (5th '
                        'keyframe of each scene). Reduces ~4,405 → ~110 samples.')
    p.add_argument('--no-anchor', action='store_true',
                   help='override dataset doscenes_anchor_only=False so the '
                        'val pkl exposes ALL 6019 samples (not just anchors).')
    p.add_argument('--use-gt-cmd', action='store_true',
                   help='ORACLE upper-bound: strip LoadLaneletCmd / '
                        'LoadDoScenesInstruction(override_cmd_from_text=True) '
                        'from the pipeline so cmd stays as the pkl-stored '
                        'gt_ego_fut_cmd (derived from GT future trajectory). '
                        'Compliance violation — diagnostic only.')
    p.add_argument('--use-speed', action='store_true',
                   help='TransFuser++-style post-hoc speed correction: rescale '
                        'each predicted step\'s offset magnitude to the speed-class '
                        'bin centre, keeping direction.')
    p.add_argument('--progress-every', type=int, default=100)
    p.add_argument('--dump-prefix', default='/tmp/eval',
                   help='per-sample npz dump path prefix '
                        '(<prefix>_with_language.npz / <prefix>_baseline.npz). '
                        'Set unique per process when running parallel evals.')
    p.add_argument('--anchor-window', type=int, default=0,
                   help='Restrict dataset to frame_idx ∈ [0..N] for each '
                        'val/test scene (N=anchor_window). N=4 keeps 5 '
                        'frames per scene so prev_bev correctly accumulates '
                        'up to the frame-4 anchor. Speedup vs full 6019 ≈ '
                        '6019/(150*(N+1)).')
    p.add_argument('--apply-stop-rule', action='store_true',
                   help='Rule v5 post-processing: when the instruction is a '
                        'HARD-STOP command (stop/halt/brake/yield/wait, no '
                        'OTHER_ACTION verb, word count <= 12, not "stop sign" '
                        'etc.) AND the ego is already nearly stationary '
                        '(history speed <= 2 m/s), zero out predicted offsets '
                        'so the trajectory holds at the anchor. Only active '
                        'in --with-language.')
    p.add_argument('--deterministic-instruction', action='store_true',
                   help='Seed numpy.random with md5(scene_token) before each '
                        'LoadDoScenesInstruction random pick so every eval '
                        'sees the same instruction for the same sample. '
                        'Use this for ensemble runs that must share inputs.')
    args = p.parse_args()

    if not args.with_language and not args.no_language:
        # default: run both passes.
        args.with_language = True
        args.no_language = True

    only_doscenes = not args.all_samples
    sys.path.insert(0, '.')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    with_lang_dump = f'{args.dump_prefix}_with_language.npz'
    baseline_dump  = f'{args.dump_prefix}_baseline.npz'

    summaries: Dict[str, Dict[str, float]] = {}
    if args.with_language:
        summaries['with_language'] = run_eval(
            cfg_path=args.config, ckpt_path=args.ckpt, gpu_id=args.gpu,
            use_language=True, max_samples=args.max_samples,
            only_doscenes=only_doscenes, anchor_only=args.anchor_only,
            no_anchor=args.no_anchor,
            use_gt_cmd=args.use_gt_cmd,
            progress_every=args.progress_every,
            dump_path=with_lang_dump,
            use_speed_correction=args.use_speed,
            deterministic_instruction=args.deterministic_instruction,
            anchor_window=args.anchor_window,
            apply_stop_rule=args.apply_stop_rule,
        )
        _print_block('with-language', summaries['with_language'])

    if args.no_language:
        summaries['baseline'] = run_eval(
            cfg_path=args.config, ckpt_path=args.ckpt, gpu_id=args.gpu,
            use_language=False, max_samples=args.max_samples,
            only_doscenes=only_doscenes, anchor_only=args.anchor_only,
            no_anchor=args.no_anchor,
            use_gt_cmd=args.use_gt_cmd,
            progress_every=args.progress_every,
            dump_path=baseline_dump,
            use_speed_correction=args.use_speed,
            keep_cmd_no_lang=args.keep_cmd_no_lang,
            deterministic_instruction=args.deterministic_instruction,
            anchor_window=args.anchor_window,
        )
        _print_block('baseline (no language)', summaries['baseline'])

    if 'with_language' in summaries and 'baseline' in summaries:
        s_inst = summaries['with_language']
        s_base = summaries['baseline']
        delta_2s = s_base['ade_2s'] - s_inst['ade_2s']
        delta_4s = s_base['ade_4s'] - s_inst['ade_4s']
        delta_6s = s_base['ade_6s'] - s_inst['ade_6s']
        print('\n' + '=' * 64)
        print('ΔADE = baseline − instruction (positive = language helps)')
        print('=' * 64)
        print(f'  ΔADE@2s = {delta_2s:+.6f}')
        print(f'  ΔADE@4s = {delta_4s:+.6f}')
        print(f'  ΔADE@6s = {delta_6s:+.6f}    <-- official metric')
        pt_keys = [f'ade_pt{i}' for i in range(1, 13)]
        if all(k in s_inst and k in s_base for k in pt_keys):
            print('  ΔADE@12pts (0.5s..6.0s):')
            vals = '  '.join(
                f'{(s_base[k] - s_inst[k]):+.3f}' for k in pt_keys)
            print(f'    {vals}')

        # Per-sample diff diagnostic — catches cases where averages round to
        # the same value but the per-sample preds actually differ.
        try:
            with_arr  = np.load(with_lang_dump)
            base_arr  = np.load(baseline_dump)
            diff = with_arr['ade_6s'] - base_arr['ade_6s']
            n_total = len(diff)
            n_same  = int(np.sum(np.isclose(diff, 0.0, atol=1e-9)))
            n_diff  = n_total - n_same
            print('\n  per-sample ade_6s diff (with-lang − baseline):')
            print(f'    n_total              = {n_total}')
            print(f'    n_identical (atol 1e-9) = {n_same}')
            print(f'    n_different          = {n_diff}')
            if n_diff > 0:
                print(f'    abs(diff): min={np.min(np.abs(diff[diff!=0])):.6f}  '
                      f'mean={np.mean(np.abs(diff)):.6f}  max={np.max(np.abs(diff)):.6f}')
                print(f'    diff sign: pos(lang better)={int((diff>0).sum())}  '
                      f'neg(lang worse)={int((diff<0).sum())}')
        except Exception as exc:
            print(f'  (per-sample diff skipped: {exc})')


if __name__ == '__main__':
    main()
