"""Stage C scale-up BIG-MLP — capacity expansion variant for Task 2.

User's original ask: "Stage C + ego_fut_mode=3→6 (v8-style routing), 預期 -0.1~0.2 m,
挑戰 2.76 m". True 3→6 mode switch requires:
  - retraining pts_bbox_head ego_fut_decoder last layer (72→144 out)
  - cmd index remap in eval (lanelet 3-class → 6-class)
  - swapping load_from from v4_resume to v8 (ckpt has 6-mode head)
which is high-risk wiring and was deferred. The underlying hypothesis the user
wanted to test is "language channel capacity is the bottleneck, not prompt
richness". Bigger TextDeltaPlanner MLP tests the same hypothesis without
breaking shape/ckpt compatibility:

  text_proj_dim   256 → 512   (4 M params)
  mlp_hidden_dim  512 → 1024  (~2-3 M params)
  ego_fut_mode    3 (unchanged, matches efp)
  fut_ts          12 (unchanged)

Output shape still [B, 3, 12, 2] → no shape juggling, drops in cleanly.

Also inherits Task 1's BN-freeze fix (via scaleup_v2) so no-lang baseline
stays bit-identical to v4_resume baseline.

Train spec: 60 ep, 4-GPU DDP, eff batch 8, lr 1e-4 cosine, grad_clip=2.
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4_resume_text_delta_scaleup_v2.py']

model = dict(
    text_delta_text_proj_dim=512,
    text_delta_hidden_dim=1024,
    text_delta_max_length=64,
)
