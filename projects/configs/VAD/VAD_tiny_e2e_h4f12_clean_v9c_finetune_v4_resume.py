"""v9c_finetune v4_resume: continue from v4 epoch_30 with fresh cosine schedule.

Vs v4:
  - load_from: shu_wei_stripped -> v4 epoch_30 (continue from v4 final ckpt)
  - lr peak: 2e-4 -> 1e-4 (lower because we're closer to convergence)
  - warmup_iters: 500 -> 100 (short warmup; weights already aligned)
  - 30 more epochs, fresh cosine 1e-4 → 1e-7

Keeps v4's recipe: anchor +/- 1 window (3,4,5), ramp 1->4, FDE 1.0.
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4.py']

# Continue from v4 ep30 (not shu_wei stripped)
load_from = 'output_v9c_finetune_v4/epoch_30.pth'

# Lower peak lr (continuing training, no need to scale up again)
optimizer = dict(lr=1e-4)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
