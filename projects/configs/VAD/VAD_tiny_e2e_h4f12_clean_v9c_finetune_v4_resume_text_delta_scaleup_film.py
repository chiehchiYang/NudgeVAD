"""Phase B scale-up: bigmlp + BN-freeze + FiLM-v1 (γ+β+α, full).

Combines the three improvements that each individually moved with-lang anchor:
  - BN-freeze fix (v2):    train-time img_backbone/img_neck/pts_bbox_head in eval()
                           → train signal cleaner → anchor 2.978 m
  - MLP capacity ×2.6:     text_proj 256→512, mlp_hidden 512→1024
                           → bigmlp anchor 2.972 m
  - FiLM modulation:       γ·ego_feats + β with γ_init=1, β_init=0, α-gated
                           → expected another -0.05 to -0.1 m if subset shows >0.27 ΔADE

Target: anchor ADE@6s with-lang < 2.90 m (beat current best bigmlp 2.972 by ≥0.07 m).

Subset sanity (Phase B-pre) must pass first; default config here uses FiLM-v1
(γ+β+α). To swap variant, override `text_delta_film_use_*` in the model dict.
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4_resume_text_delta_scaleup_v2.py']

model = dict(
    # bigmlp capacity
    text_delta_text_proj_dim=512,
    text_delta_hidden_dim=1024,
    text_delta_max_length=64,
    # FiLM modulation — v4 (γ+β, NO α gate) — subset winner +0.461 ΔADE
    # α gate is removed because γ_init=1 + β_init=0 + MLP-last-init=0 already
    # guarantees first-iter == baseline; α was a redundant bottleneck that
    # blocked gradient flow to γ/β projections. Confirmed by 4-variant subset
    # ablation (v4 beats v1/v2/v3 on all metrics).
    text_delta_use_film=True,
    text_delta_film_use_gamma=True,
    text_delta_film_use_beta=True,
    text_delta_film_use_alpha=False,
    text_delta_film_use_concat=False,
)
