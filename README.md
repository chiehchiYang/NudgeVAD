# NudgeVAD — Instructed Driving with Frozen VAD + LLaVA + FiLM Adapter

NudgeVAD is a planner that adds **natural-language instruction conditioning** on top of a frozen [VAD-Tiny](https://github.com/hustvl/VAD) trunk and a frozen [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA) vision-language encoder. A small **TextDeltaPlanner** adapter (γ/β FiLM modulation, ~2M trainable params) produces a residual offset on VAD's ego trajectory.

Submitted to the **CVPR 2026 doScenes Instructed-Driving Challenge** ([mi3-lab.github.io/doScenes_challenge](https://mi3-lab.github.io/doScenes_challenge)).

## Key result

**Val anchor N=150 (frame_idx==4, doScenes no-cmd line, random cmd matching training):**

| Method | a@1s | a@2s | a@3s | a@4s | a@5s | a@6s ↓ | FDE ↓ | ΔADE@6s ↑ |
|---|---|---|---|---|---|---|---|---|
| VAD-Tiny baseline (Stage 1 ep60 no-cmd) | 0.360 | 1.100 | 1.605 | 2.275 | 2.923 | 3.590 | 7.333 | — |
| + Stage 1 continue +60 ep trunk (no lang) | 0.337 | 1.088 | 1.561 | 2.031 | 2.526 | 3.118 | 6.517 | +0.472 |
| + plain language prompt | 0.377 | 1.008 | 1.589 | 2.066 | 2.577 | 3.170 | 6.547 | +0.420 |
| + rich prompt (intent + refs) | 0.363 | 0.967 | 1.524 | 1.991 | 2.501 | 3.097 | 6.493 | +0.493 |
| + BN-freeze fix (v2) | 0.375 | 1.031 | 1.635 | 2.120 | 2.634 | 3.229 | 6.613 | +0.361 |
| + MLP capacity ×2.6 (bigmlp) | 0.369 | 0.912 | 1.418 | 1.863 | 2.367 | 2.961 | 6.341 | +0.629 |
| **NudgeVAD (FiLM-v4)**  | **0.351** | **0.846** | **1.305** | **1.731** | **2.221** | **2.806** | **6.148** | **+0.784** |
| + Stop override  ⭐ | 0.348 | 0.836 | 1.290 | 1.713 | 2.197 | 2.774 | 6.071| 0.816 |


ΔADE@6s = cumulative improvement vs VAD-Tiny baseline (no-cmd Stage 1 ep60). NudgeVAD's own same-model own-ΔADE (with-language minus no-language, cmd-matched random one-hot) is **+0.360 m** at a@6s. The official challenge "history-only" baseline reports ΔADE = −0.096 on test — NudgeVAD is, to our knowledge, the first reported method that produces a clearly **positive** ΔADE under doScenes's no-future-leakage protocol.

## Architecture

```
                  ┌─────────────────────────────────────┐
   instruction →  │  LLaVA-1.5-7B (frozen + LoRA r=16) │ → text_vec [B, 4096]
                  └─────────────────────────────────────┘
                                                              │
   6-cam images → VAD-Tiny BEV → ego_feats [B, 256] ─────────┤
   ego_history ─┘                                             │
                                                              ▼
                            ┌──────────────────────┐
                            │ TextDeltaPlanner     │
                            │   FiLM-v4 modulation │
                            │     γ = γ_proj(text) │
                            │     β = β_proj(text) │
                            │ ego_feats * γ + β    │
                            │ → 3-layer MLP        │
                            │ → delta [B,3,12,2]   │
                            └──────────────────────┘
                                                              │
   VAD ego planner head → ego_fut_preds [B,3,12,2] ──────────┤
                                                              ▼
                                  output = ego_fut_preds + delta
                                          (γ_init=1, β_init=0, MLP last layer zero
                                           ⇒ first-iter output identical to baseline)
```

**Design principles:**
- **Frozen base** — VAD-Tiny ego planner + LLaVA both frozen. Only the FiLM adapter + LoRA on LLaVA's q/v projections are trainable.
- **Init-safe** — At iteration 0, γ ≡ 1, β ≡ 0, MLP last layer is zero, so the adapter output is identical to the baseline. Training can only **improve**, not regress.
- **No future leakage** — VAD's stock `ego_fut_cmd` channel is derived from the GT future trajectory; we replace it with `ForceCmdNeutral(mode='random')` so the model never sees direction information leaked from future GT.

See [projects/mmdet3d_plugin/VAD/text_delta_planner.py](projects/mmdet3d_plugin/VAD/text_delta_planner.py) for the adapter implementation.


## Setup

### 1. Conda environment

```bash
conda create -n nudgevad python=3.10 -y
conda activate nudgevad
pip install -r requirements.txt
# Install mmcv-full / mmdet / mmdet3d per the original VAD instructions:
#   https://github.com/hustvl/VAD#installation
```

### 2. Data

The repository expects this directory layout (none of these files are committed):

```
data/nuscenes/
├── v1.0-trainval/                # standard nuScenes
├── v1.0-test/                    # for test-set submission
├── samples/  sweeps/  maps/
├── vad_nuscenes_h4f12_clean_train.pkl       # h4f12 = history-4 future-12
├── vad_nuscenes_h4f12_clean_val.pkl
├── vad_nuscenes_h4f12_infos_temporal_test.pkl
└── scene_token_to_name.json

third_party/doScenes/
├── Annotations/         # cloned from https://github.com/rossgreer/doScenes
└── paths.txt            # NUSCENES_ROOT=...  DOSCENES_ANNOTATIONS=...
                         # (start from paths.txt.example)
```

To build the h4f12 pkl files from raw nuScenes use the [VAD data converter](https://github.com/hustvl/VAD#data-preparation) with `history_steps=4, future_steps=12`. To fetch doScenes annotations:

```bash
git clone https://github.com/rossgreer/doScenes third_party/doScenes
cp third_party/doScenes/paths.txt.example third_party/doScenes/paths.txt
# edit paths.txt to point at your nuScenes root + doScenes/Annotations
```

### 3. Pretrained weights

**VAD-Tiny init** (`VAD_tiny_e2e_h4f12.pth`) — Stage 1 trains from this file as `load_from`. It is derived from the official [VAD model zoo](https://github.com/hustvl/VAD#model-zoo) `VAD_tiny_e2e.pth` by dropping 4 shape-mismatched keys (h2f6 → h4f12 head). Reproduce:

```bash
wget <VAD_zoo_URL>/VAD_tiny_e2e.pth -O ckpts/VAD_tiny_e2e.pth
python tools/strip_ckpt_for_v9c_finetune.py \
  --src ckpts/VAD_tiny_e2e.pth --dst ckpts/VAD_tiny_e2e_h4f12.pth
```

See [ckpts/README.md](ckpts/README.md) for the full provenance chain (zoo → stripped → Stage 1 → ablations + NudgeVAD).

**LLaVA-1.5-7B** — downloaded automatically via HuggingFace cache on first run. Set `HF_HOME` to control cache location.

### 4. Our trained checkpoints (skip training)

All 7 checkpoints in our val table are released under [`ckpts/`](ckpts/) (4.1 GB total). See [ckpts/README.md](ckpts/README.md) for the per-file mapping (ckpt → val a@6s → training config). The main NudgeVAD checkpoint is [`ckpts/nudgevad_film_v4_ep60.pth`](ckpts/nudgevad_film_v4_ep60.pth) (662 MB). Integrity check via `cd ckpts && sha256sum -c SHA256SUMS`.

> The `.pth` files are NOT committed to git (size). Download links to the cloud-hosted copies live in [ckpts/README.md](ckpts/README.md).

## Training

NudgeVAD is trained in two stages: a no-cmd VAD-Tiny trunk on doScenes anchors (Stage 1), then a frozen-trunk adapter on top (Stage 2). All configs use `ForceCmdNeutral(mode='random')` so the planner never sees future-derived direction hints.

**Default: 4 × 24 GB GPUs.** Per-GPU memory is ~17.5 GB, so the recipe fits comfortably on 4 RTX-4090s and scales linearly up to 8 GPUs. With 4 GPUs the effective batch is `4 × samples_per_gpu (2) × cumulative_iters (2) = 16` (vs the internal 32 on 8 GPUs); we observed no notable change in the final ADE numbers. To match the 8-GPU effective batch on 4 GPUs, raise `cumulative_iters` to 4 in `--cfg-options`. All commands assume the conda env is activated; adjust `--work-dir` as you wish.

### Stage 1 — VAD-Tiny baseline (no-cmd, 90 ep)

The Stage 1 baseline uses `ForceCmdNeutral(random)` and the doScenes anchor-only train split (frame_idx==4). It is the reference no-language model.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=28509 \
  bash tools/dist_train.sh \
    projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_anchor_only_90ep.py 4 \
    --work-dir output_v9c_nocmd_anchor_only_90ep
```

Recommended checkpoint: `epoch_60.pth` (validation usually plateaus around 60 ep). This is the file used as **"VAD-Tiny baseline (Stage 1 ep60 no-cmd)"** in our results table.

### Stage 1 continue — extra 60 ep on the trunk (compute-fair baseline)

If you want a no-language baseline matched in compute to NudgeVAD (Stage 2 runs 60 ep), continue training the trunk for 60 more epochs from the Stage 1 ep60 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=28509 \
  bash tools/dist_train.sh \
    projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_continue_60ep.py 4 \
    --work-dir output_v9c_nocmd_continue_60ep
```

This is the **"+ Stage 1 continue +60 ep trunk (no lang)"** row in the table. It controls for "did the +0.92m ΔADE come from architecture or from compute?".

### Stage 2 — NudgeVAD (FiLM-v4) ⭐

NudgeVAD freezes the Stage 1 ep60 trunk and trains the FiLM-v4 TextDeltaPlanner + LoRA on LLaVA's q/v projections. 60 ep, ~8 h on 4 × 24 GB GPUs (~4 h on 8 GPUs).

```bash
# Make sure load_from points at your Stage 1 ep60 checkpoint
# (see projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py).
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=28509 \
  bash tools/dist_train.sh \
    projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py 4 \
    --work-dir output_v9c_nocmd_nudgevad_60ep
# Optional: preserve the 8-GPU effective batch (32) on 4 GPUs by appending:
#   --cfg-options optimizer_config.cumulative_iters=4
```

Use `epoch_60.pth` for evaluation and submission.

### Ablations (optional)

Four single-axis ablations to isolate the contribution of each NudgeVAD component:

| Variant | Config | What's tested |
|---|---|---|
| `plain` | `VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_plain_60ep.py` | TextDeltaPlanner with plain instruction prompt (no intent / reference structure) |
| `rich` | `VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_rich_60ep.py` | Plain + intent type + static/dynamic reference flags |
| `v2` | `VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_v2_60ep.py` | Rich + BN-freeze fix on text projection |
| `bigmlp` | `VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_bigmlp_60ep.py` | v2 + MLP capacity ×2.6 (text_proj 512, mlp_hidden 1024) |

All four are trained the same way as Stage 2 with their respective configs.

## Inference and self-evaluation (on val)

### Per-scene val eval (N=150, original convention)

`tools/eval_doscenes_local.py` runs the model on the val set, computes ADE/FDE/ΔADE, and dumps per-sample npz arrays. It supports the optional **rule v5 stop override** (see paper).

```bash
python tools/eval_doscenes_local.py \
  --config projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py \
  --ckpt output_v9c_nocmd_nudgevad_60ep/epoch_60.pth \
  --gpu 0 --anchor-window 4 \
  --with-language --no-language \
  --dump-prefix /tmp/eval_nudgevad
# Add `--apply-stop-rule` to enable rule v5.
```

Output: `/tmp/eval_nudgevad_with_language.npz`, `/tmp/eval_nudgevad_baseline.npz`, and a printed ADE/FDE/ΔADE table.

### Per-(scene, instruction) pair val eval (N=310, aligns with test convention)

`tools/eval_doscenes_pairs.py` enumerates every doScenes instruction for each val anchor and forwards the model once per (scene, instruction) pair. This matches the row granularity of the official 493-row test submission.

```bash
python tools/eval_doscenes_pairs.py \
  --config projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py \
  --ckpt   output_v9c_nocmd_nudgevad_60ep/epoch_60.pth \
  --gpu 0 --with-language --no-language \
  --dump-prefix /tmp/eval_pairs_nudgevad
```

To re-run all 7 methods in parallel on 8 GPUs, see the example `run_pairs_eval.sh` snippet at the end of this README.

## Test-set submission

The official challenge accepts predictions on the **150 v1.0-test scenes** in the 26-column mi3-lab format:

```
sample_token, instruction, x1, y1, x2, y2, ..., x12, y12
```

One row per scene; 12 cumulative future positions at 0.5 s intervals in the ego frame at the anchor (X forward, Y left). See [submissions/README.md](submissions/README.md) for the released CSVs and self-eval numbers.

### Step 1 — Run inference on the test set (no GT touch)

`tools/predict_test150.py` is **inference-only**: it does NOT read `gt_ego_fut_trajs` from the test pkl and does NOT compute ADE/FDE on the test set. Its only job is to forward the model on the 150 v1.0-test anchors and write the two submission CSVs in 26-col mi3-lab format.

The script walks 5 frames per scene to keep VAD's stateful `prev_bev` temporal queue correct, then runs two forwards at each anchor (with-language using the first doScenes instruction, no-language with random cmd matching training).

```bash
# Override config to point the val pipeline at the test pkl (metadata only —
# lidar2ego rotation, history trajectory; the future field is never read).
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

Rule v5 (HARD_STOP keyword + `no OTHER_ACTION verb` + `≤ 12 words` + `history speed ≤ 2 m/s`) fires on 3 scenes:

```
- "stop at red light"
- "Wait here"
- "stop at the red light behind the crosswalk"
```

Outputs (committed to this repo):

- [`submissions/test150_NudgeVAD_FiLMv4_rule_v5.csv`](submissions/test150_NudgeVAD_FiLMv4_rule_v5.csv) ⭐ **main submission**
- [`submissions/test150_NudgeVAD_FiLMv4_no_language.csv`](submissions/test150_NudgeVAD_FiLMv4_no_language.csv) — baseline

### Step 2 — Validate format

```bash
head -1 submissions/test150_NudgeVAD_FiLMv4_rule_v5.csv
# sample_token,instruction,x1,y1,x2,y2,...,x12,y12
wc -l submissions/test150_NudgeVAD_FiLMv4_*.csv
# 151 (header + 150 rows) each
```

### Step 3 — Upload

Upload `submissions/test150_NudgeVAD_FiLMv4_rule_v5.csv` to the [challenge leaderboard](https://mi3-lab.github.io/doScenes_challenge). **All test ADE/FDE numbers come from the official leaderboard, never from local computation** — this repository contains no code that scores predictions against test ground-truth.

## Reproducing the full ablation table

Launch all 7 methods × 2 passes in parallel on 8 GPUs (each method takes ~5 minutes per pass):

```bash
#!/bin/bash
cd /path/to/repo
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nudgevad
export HF_HUB_OFFLINE=1

launch() {
  local gpu=$1 key=$2 cfg=$3 ckpt=$4 extra=${5:-}
  CUDA_VISIBLE_DEVICES=$gpu python tools/eval_doscenes_pairs.py \
    --config $cfg --ckpt $ckpt --gpu 0 \
    --with-language --no-language \
    --dump-prefix /tmp/eval_pairs_$key $extra \
    > /tmp/$key.log 2>&1 &
}

launch 0 stage1_ep60 projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_anchor_only_90ep.py output_v9c_nocmd_anchor_only_90ep/epoch_60.pth
launch 1 continue_60 projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_continue_60ep.py    output_v9c_nocmd_continue_60ep/epoch_60.pth
launch 2 plain      projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_plain_60ep.py  output_v9c_nocmd_ablation_plain_60ep/epoch_60.pth
launch 3 rich       projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_rich_60ep.py   output_v9c_nocmd_ablation_rich_60ep/epoch_60.pth
launch 4 v2         projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_v2_60ep.py     output_v9c_nocmd_ablation_v2_60ep/epoch_60.pth
launch 5 bigmlp     projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_bigmlp_60ep.py output_v9c_nocmd_ablation_bigmlp_60ep/epoch_60.pth
launch 6 nudgevad   projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py        output_v9c_nocmd_nudgevad_60ep/epoch_60.pth
wait
```

Then aggregate into a markdown table by reading the 14 npz files (see `tools/eval_doscenes_pairs.py` output format).

## Repository layout

```
.
├── projects/
│   ├── configs/
│   │   ├── _base_/                  # mmdetection3d shared schedules + runtime
│   │   └── VAD/                     # all training configs (this paper's tree)
│   └── mmdet3d_plugin/              # custom modules: VADLLaVA, TextDeltaPlanner,
│                                    # ForceCmdNeutral, LoadDoScenesInstruction, …
├── tools/
│   ├── train.py                     # mmdet3d-style trainer
│   ├── test.py                      # generic mmdet3d test
│   ├── eval_doscenes_local.py       # per-scene VAL ADE / FDE / ΔADE (writes npz)
│   ├── eval_doscenes_pairs.py       # per-(scene, instruction) pair VAL eval
│   ├── predict_test150.py           # inference-only 150-row test submission writer
│   ├── strip_ckpt_for_v9c_finetune.py
│   ├── dist_train.sh
│   └── dist_test.sh
├── third_party/
│   └── doScenes/
│       ├── dataloader.py            # **official** doScenes per-pair dataloader (reference)
│       └── paths.txt.example
├── ckpts/                           # 7 trained checkpoints (4.1 GB total).
│   ├── README.md                    # per-file mapping + SHA256SUMS
│   ├── SHA256SUMS
│   └── *.pth                        # NOT tracked by git → uploaded to cloud,
│                                    # download links live in ckpts/README.md
├── submissions/                     # 150-row test submissions (~40-45 KB each)
│   ├── README.md
│   ├── test150_NudgeVAD_FiLMv4_rule_v5.csv      ⭐ main: with-lang + rule v5
│   ├── test150_NudgeVAD_FiLMv4_no_language.csv     baseline (own-ΔADE)
│   └── archive_493row/                          # earlier per-pair CSVs (reference)
├── requirements.txt
├── LICENSE                          # Apache 2.0 (inherits from VAD)
└── README.md
```

## Citation

If you use NudgeVAD, please cite:

```bibtex
@inproceedings{nudgevad2026,
  title  = {NudgeVAD: Instructed Driving with Frozen VAD and Language-Conditioned FiLM Adapter},
  author = {Yang, Chieh-Chi and Chen, Yu-Hsiang and Chen, Yi-Ting},
  booktitle = {CVPR 2026 doScenes Instructed-Driving Challenge},
  year   = {2026},
}
```

And please also cite the projects we build on:

- [VAD](https://github.com/hustvl/VAD) — base planner trunk
- [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) — frozen vision-language encoder
- [doScenes](https://github.com/rossgreer/doScenes) — instruction annotations
- [doScenes Challenge Starter](https://github.com/vishnu-anirudh/multimodal-instruction-driving) — submission format reference

## License

This codebase inherits VAD's Apache 2.0 license. See [LICENSE](LICENSE).
