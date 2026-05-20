"""Stage C scale-up + RICH PROMPT 變體.

vs scaleup (base):
  - text_delta_rich_prompt=True  → LLaMA 看到 instruction + intent type
                                    + static_ref + dynamic_ref 四件事
  - text_delta_max_length 64 → 96 (rich prompt 長一點)

其他超參完全一樣 (60 ep, 4-GPU DDP, eff batch 8, lr 1e-4 cosine, clip=2)。
Compliance-safe:所有額外 meta 都來自 doScenes annotation,沒 future leak。
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4_resume_text_delta_scaleup.py']

model = dict(
    text_delta_rich_prompt=True,
    text_delta_max_length=96,
)
