"""Compute-fair counterpart to Stage 2.

Setup:
  - load_from: Stage 1 ep60 (same starting point as Stage 2)
  - 60 epochs of pure VAD trunk training (NO language, NO adapter)
  - Same random-cmd no-cmd pipeline as Stage 1

Purpose:
  Compare against Stage 2 (Stage 1 ep60 + 60 ep adapter):
    - If this run's ade@6s >= 2.806 (Stage 2)
        → adapter has independent value beyond more trunk training
    - If this run's ade@6s < 2.806
        → trunk continuation subsumes language adapter (Plan B repeated)
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_nocmd_anchor_only_90ep.py']

# Override load_from to start from Stage 1 ep60
load_from = 'output_v9c_nocmd_anchor_only_90ep/epoch_60.pth'

# 60 more epochs (fresh cosine schedule)
total_epochs = 60
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=10, max_keep_ckpts=7)
evaluation = dict(interval=total_epochs + 1)
