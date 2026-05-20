"""Fine-tune Stage 2 ep60 on stop-emphasis train pkl.

Stop samples (final motion < 5m, low speed at end) are replicated 4x in
the train pkl → ~50% of anchor samples are stop scenarios (vs 16% original).
Adapter learns to map "stop"/"slow"/"yield" language → reduced trajectory
magnitude.

Setup:
  - load_from: Stage 2 ep60 (current best NudgeVAD)
  - train ann_file: stop-replicated train pkl
  - 30 ep fine-tune adapter
  - all other settings inherited from Stage 2 config

Target: improve test "stop" class ade@6s (currently 5.81m, language hurts -0.82m)
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py']

load_from = 'output_v9c_nocmd_nudgevad_60ep/epoch_60.pth'

# Override train pkl to stop-emphasis version
data = dict(
    train=dict(
        ann_file='data/nuscenes/vad_nuscenes_h4f12_infos_temporal_train_stop_emphasis.pkl',
    ),
)

# Shorter fine-tune: 30 ep
total_epochs = 30
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=10, max_keep_ckpts=4)
evaluation = dict(interval=total_epochs + 1)

# Lower lr for fine-tune (adapter already partially trained)
optimizer = dict(lr=5e-5)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear', warmup_iters=50, warmup_ratio=1.0/3,
    min_lr_ratio=1e-3,
)
