# Pretrained Checkpoints

7 trained checkpoints reproducing the val per-(scene, instruction) pair results in the top-level [README.md](../README.md). Each file is `epoch_60.pth` from the corresponding training run.

| File | Size | Method | Val a@6s | Train config |
|---|---|---|---|---|
| `stage1_baseline_ep60.pth` | 463 MB | VAD-Tiny baseline (Stage 1 ep60 no-cmd) | 3.545 | [`nocmd_anchor_only_90ep`](../projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_anchor_only_90ep.py) |
| `stage1_continue_60ep.pth` | 463 MB | + Stage 1 continue +60 ep trunk (no lang) | 3.125 | [`nocmd_continue_60ep`](../projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_continue_60ep.py) |
| `ablation_plain_ep60.pth` | 642 MB | + plain language prompt | 3.197 | [`nocmd_ablation_plain_60ep`](../projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_plain_60ep.py) |
| `ablation_rich_ep60.pth` | 642 MB | + rich prompt (intent + refs) | 3.110 | [`nocmd_ablation_rich_60ep`](../projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_rich_60ep.py) |
| `ablation_v2_ep60.pth` | 642 MB | + BN-freeze fix (v2) | 3.282 | [`nocmd_ablation_v2_60ep`](../projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_v2_60ep.py) |
| `ablation_bigmlp_ep60.pth` | 662 MB | + MLP capacity ×2.6 (bigmlp) | 2.910 | [`nocmd_ablation_bigmlp_60ep`](../projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_ablation_bigmlp_60ep.py) |
| **`nudgevad_film_v4_ep60.pth`** | **662 MB** | **NudgeVAD (FiLM-v4)** ⭐ | **2.626** | [`nocmd_nudgevad_60ep`](../projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py) |


## Provenance

All checkpoints in this directory descend from the **official VAD-Tiny pretrained weights** from the [VAD model zoo](https://github.com/hustvl/VAD#model-zoo) (`VAD_tiny_e2e.pth`, history=2, future=6).

### Provenance chain

```
VAD model zoo:  VAD_tiny_e2e.pth          (h2f6, ~455 MB, NOT in this repo)
                       │
                       │  tools/strip_ckpt_for_v9c_finetune.py
                       │  drops 4 keys whose shapes mismatch h4f12:
                       │    pts_bbox_head.traj_branches.0.4.{weight,bias}
                       │      ([12,512] → [24,512])
                       │    pts_bbox_head.ego_fut_decoder.4.{weight,bias}
                       │      ([36,512] → [72,512])
                       ▼
shu_wei_stripped_for_v9c.pth   (h4f12-compatible init, ~455 MB,
                                NOT in this repo — produced locally)
                       │
                       │  4-GPU DDP, 90 ep on doScenes anchor-only train pkl,
                       │  ForceCmdNeutral(random), cosine LR, ~20 h on 4 × RTX 4090
                       │  (the released ckpts were trained on 8 × RTX 4090
                       │  with samples_per_gpu=2 cumulative_iters=2;
                       │  default-4-GPU recipe halves the effective batch.)
                       ▼
stage1_baseline_ep60.pth       ← we checkpoint ep60 of the 90-ep run
   │
   │── stage1_continue_60ep.pth        (continue Stage 1 trunk +60 ep)
   │── ablation_plain_ep60.pth          \\
   │── ablation_rich_ep60.pth            \\  All 5 use stage1_baseline_ep60
   │── ablation_v2_ep60.pth               >─ as `load_from`; only the adapter
   │── ablation_bigmlp_ep60.pth          //  + LoRA layers are trained for 60 ep.
   └── nudgevad_film_v4_ep60.pth        //
```

### Reproducing the init

```bash
# 1. Download VAD-Tiny weights from the VAD model zoo
wget <VAD_zoo_URL>/VAD_tiny_e2e.pth -O ckpts/VAD_tiny_e2e.pth

# 2. Strip shape-mismatched keys so the ckpt loads into our h4f12 model
python tools/strip_ckpt_for_v9c_finetune.py \
  --src ckpts/VAD_tiny_e2e.pth \
  --dst ckpts/shu_wei_stripped_for_v9c.pth

# 3. Finetune Stage 1
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=28509 \
  bash tools/dist_train.sh \
    projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_anchor_only_90ep.py 4 \
    --work-dir output_v9c_nocmd_anchor_only_90ep
# → output_v9c_nocmd_anchor_only_90ep/epoch_60.pth == stage1_baseline_ep60.pth
```

### Training notes

- All training uses `ForceCmdNeutral(mode='random')` to overwrite the `ego_fut_cmd` channel with a random one-hot, removing the no-future-leakage violation in VAD's stock cmd derivation.
- The 4 ablations + NudgeVAD freeze the Stage 1 trunk and only train the FiLM/LoRA layers (~2 M params). See [main README](../README.md) for hyperparameters.

## Usage

```bash
# Evaluate NudgeVAD on val (per-(scene, instruction) pair)
python tools/eval_doscenes_pairs.py \
  --config projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py \
  --ckpt ckpts/nudgevad_film_v4_ep60.pth \
  --gpu 0 --with-language --no-language \
  --dump-prefix /tmp/eval_pairs_nudgevad

# Generate submission CSV
python tools/test_doscenes.py \
  --config projects/configs/VAD/VAD_tiny_e2e_h4f12_clean_v9c_nocmd_nudgevad_60ep.py \
  --ckpt ckpts/nudgevad_film_v4_ep60.pth \
  --with-language \
  --out submissions/my_submission.csv
```




