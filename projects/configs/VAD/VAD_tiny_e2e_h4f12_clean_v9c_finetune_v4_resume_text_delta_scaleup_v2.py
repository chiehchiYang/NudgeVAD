"""Stage C scale-up V2 — fix no-lang 0.105 m artifact.

Same as scaleup (Plain) but with the BN-freeze fix:
  - VADLLaVA.train() now calls .eval() on img_backbone/img_neck/pts_bbox_head
    whenever text_delta_planner_only=True. This locks BN running stats so the
    frozen base stays bit-identical to the load_from v4_resume baseline.

Expectation: no-lang anchor @6s should now match baseline 3.146 m (currently
3.251 m). With-lang gets the same shift down → anchor @6s ≈ 2.90 m, ΔADE
relative to no-lang stays ≈ +0.24 m, but absolute number breaks Stage C
plain ceiling.

Train spec identical to scaleup: 60 ep, 4-GPU DDP, eff batch 8, lr 1e-4
cosine, grad_clip max_norm=2, cumulative_iters=2.
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4_resume_text_delta_scaleup.py']

# No model changes — fix is in VADLLaVA.train() override. Re-declare empty
# model dict to ensure inheritance chain is preserved unchanged.
