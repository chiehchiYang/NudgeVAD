# Submission CSVs (doScenes test, N=150 per-scene)

Two 150-row submission CSVs in the **26-column mi3-lab format**:

```
sample_token, instruction, x1, y1, x2, y2, ..., x12, y12
```

Predictions are 12 cumulative future positions at 0.5 s intervals (0.5 s, 1.0 s, …, 6.0 s), in the **ego frame at the anchor** (X forward, Y left). One row per scene (150 total v1.0-test scenes); for the 23 scenes without doScenes annotation, the `instruction` field is empty.

| File | rows | Description |
|---|---|---|
| **`test150_NudgeVAD_FiLMv4_rule_v5.csv`** ⭐ | 150 | **Main submission**. NudgeVAD (FiLM-v4) with-language + rule v5 stop override. For each scene the model is conditioned on its first doScenes instruction; rule v5 zeros the trajectory when (a) instruction matches HARD_STOP keywords (stop/halt/brake/yield/wait), (b) no OTHER_ACTION verb is present, (c) ≤ 12 words, (d) history speed ≤ 2 m/s. |
| `test150_NudgeVAD_FiLMv4_no_language.csv` | 150 | Baseline (no-language) reference. Same checkpoint, instruction blanked, random cmd from `ForceCmdNeutral(mode='random')` matching the training distribution. |

## No local test scoring

This repository **does not** compute ADE / FDE on the test set. All test metrics come from the official leaderboard after submission. The `tools/eval_doscenes_local.py` and `tools/eval_doscenes_pairs.py` scripts compute ADE on the **val** pkl only — pointing them at the test pkl is documented as a violation in their docstrings. The inference script that produces these CSVs (`tools/predict_test150.py`) is forward-only and never reads `gt_ego_fut_trajs`.

## How the two CSVs were produced

```bash
# (one-time) override config that points the val pipeline at the test pkl —
# only metadata (lidar2ego, history trajectory) is read; future positions
# from the pkl are never touched.
cat > projects/configs/VAD/_test_pkl_override.py <<'EOF'
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py']
data = dict(val=dict(
    ann_file='data/nuscenes/vad_nuscenes_h4f12_infos_temporal_test.pkl',
))
EOF

CUDA_VISIBLE_DEVICES=0 python tools/predict_test150.py \
  --config projects/configs/VAD/_test_pkl_override.py \
  --ckpt ckpts/nudgevad_film_v4_ep60.pth \
  --apply-stop-rule \
  --out-dir submissions
```

The script walks 5 frames per scene (frame_idx 0..4) to keep VAD's stateful `prev_bev` correct, then runs two forwards at each anchor (with-lang using the first doScenes instruction, no-lang with random cmd kept from the pipeline). Rule v5 is applied post-hoc only when `--apply-stop-rule` is set.

## Rule v5 fires (3 scenes on test)

```
- "stop at red light"
- "Wait here"
- "stop at the red light behind the crosswalk"
```

## Archived 493-row submissions

Earlier we generated per-(scene, instruction) pair submissions (493 rows). They live in `archive_493row/` for reference but the 150-row per-scene aggregation is the format we upload to the leaderboard.
