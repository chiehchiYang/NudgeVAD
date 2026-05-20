"""v9c_finetune v4: v3 + anchor window ±1 + more aggressive FDE loss.

Vs v3:
  - Train data: anchor only (700) -> anchor ±1 (3 frames, 2100 samples) → 3×
  - plan_step_weight_end: 2.0 -> 4.0 (steeper ramp)
  - loss_plan_fde_weight: 0.5 -> 1.0 (FDE term equal to plan_reg per-step)

Eval still anchor-only (frame_idx==4) for fair comparison.
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c.py']

load_from = 'ckpts/shu_wei_stripped_for_v9c.pth'

model = dict(
    pts_bbox_head=dict(
        plan_step_weight_end=4.0,    # aggressive ramp 1.0 -> 4.0
        loss_plan_fde_weight=1.0,    # FDE term 1x plan_reg
    ),
)

data = dict(
    train=dict(
        doscenes_anchor_only=False,
        doscenes_anchor_frames=[3, 4, 5],   # anchor +/- 1
    ),
    # val/test stay anchor-only (eval at frame_idx==4)
)
