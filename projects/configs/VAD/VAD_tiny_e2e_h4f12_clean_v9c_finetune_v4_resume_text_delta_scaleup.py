"""Stage C scale-up: TextDeltaPlanner on full anchor±1 (2100 samples).

Inherits the overfit config (frozen VAD planner + LoRA q,v + Linear+MLP+alpha
+ ego_feat_dim=512 eager build) but switches data source from 50-sample
overfit pkl to full anchor±1 (frame_idx in [3,4,5]) train pkl.

Training spec (matched to v8 effective batch where possible):
  - 4 GPU DDP (GPU 4,5,6,7 same NUMA)
  - per-GPU batch = 1, grad_accum = 2 → effective batch = 8
  - 60 epochs, lr 1e-4 cosine, grad_clip max_norm=2 (same as v2/v8)
  - ckpt every 5 epochs (12 ckpts saved for batch-eval curve)

Frozen / Trainable params:
  - FROZEN: img_backbone, img_neck, pts_bbox_head, LLaMA backbone,
            vision_tower, multi_modal_projector
  - TRAINABLE (~10-12M): text_delta_planner.text_proj, .mlp, .alpha
                         + LoRA q_proj/v_proj on LLaMA (128 adapters)

Pass criterion (vs v4_resume baseline anchor @6s = 2.76 m):
  - with-lang anchor @6s should NOT exceed 2.76 m (alpha=0 init guarantees this
    at epoch 0; we verify it holds at epoch 60)
  - ΔADE@6s anchor (no-lang baseline − with-lang) ≥ +0.05 m,
    ideally trending upward over epochs.
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4_resume_text_delta_overfit.py']

# === Data: full anchor±1 (2100 samples) ===
data_root = 'data/nuscenes/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_train.pkl',
        doscenes_anchor_only=False,
        doscenes_anchor_frames=[3, 4, 5],   # anchor ±1
    ),
    val=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_val.pkl',
        doscenes_anchor_only=True,           # eval at frame_idx==4
        doscenes_anchor_frames=None,
    ),
    test=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_val.pkl',
        doscenes_anchor_only=True,
        doscenes_anchor_frames=None,
    ),
)

# === Optimizer: same frozen recipe + grad_accum=2 for eff batch 8 ===
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=2,            # 4 GPU × 1 sample × 2 accum = 8
    grad_clip=dict(max_norm=2, norm_type=2),
)

# === DDP: legacy LLaVA branch params (_planning_adapters / _llava_plan_head)
#         are built but unused when text_delta_planner_only=True. Tell DDP
#         to skip the unused-param check on backward. ===
find_unused_parameters = True

# === Schedule: 60 ep, ckpt every 5 ===
total_epochs = 60
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=5, max_keep_ckpts=total_epochs // 5 + 1)
evaluation = dict(interval=total_epochs)   # no in-training eval (mmcv bug)
log_config = dict(interval=50)
