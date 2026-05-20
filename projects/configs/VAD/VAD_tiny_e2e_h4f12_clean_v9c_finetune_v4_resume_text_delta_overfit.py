"""Stage C overfit sanity — TextDeltaPlanner only (no Q-Former, no LLaVA generate).

Frozen: 全部 VAD + LLaMA backbone + vision_tower
Trainable (~10-12 M):
  - LoRA r=16 / alpha=32 on LLaMA q_proj, v_proj
  - TextDeltaPlanner.text_proj  (Linear 4096→256)
  - TextDeltaPlanner.mlp        (Linear → LN → GELU → Linear)
  - TextDeltaPlanner.alpha      (1 param, init=0)

50-sample overfit sanity (same pkl for train + val) on GPU 4。
Pass = loss_plan_reg < 0.05 + alpha.item() > 0.01。
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4_resume_llava.py']

# === Override: enable text-delta planner, disable original LLaVA Q-Former ===
model = dict(
    type='VADLLaVA',
    text_delta_planner_enabled=True,
    text_delta_planner_only=True,        # skip original LLaVA Q-Former + generate
    text_delta_text_proj_dim=256,
    text_delta_hidden_dim=512,
    text_delta_max_length=64,
    text_delta_alpha_init=0.0,
    # 關掉 original LLaVA branch 內的所有 forward 路徑 + loss
    llava_use_planning_adapter=False,
    llava_use_projector=False,
    llava_replace_ego_fut_preds=False,
    llava_use_plan_constraint_loss=False,
    llava_plan_loss_weight=0.0,
    llava_qa_loss_weight=0.0,
    # 保留 LoRA on LLaMA q,v (用於 TextDeltaPlanner 的 text encode)
    llava_enabled=True,                   # 觸發 _lazy_init_llava (含 LoRA)
    llava_use_lora=True,
    llava_lora_r=16,
    llava_lora_alpha=32,
    llava_lora_dropout=0.05,
    llava_lora_target_modules=['q_proj', 'v_proj'],
    llava_gradient_checkpointing=True,    # 24 GB GPU
    llava_freeze=True,
)

# === Data: overfit50 pkl,train + val 同一份 force memorization ===
data_root = 'data/nuscenes/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_train_overfit50.pkl',
        doscenes_anchor_only=True,
        doscenes_anchor_frames=None,
    ),
    val=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_train_overfit50.pkl',
        doscenes_anchor_only=True,
        doscenes_anchor_frames=None,
    ),
    test=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_train_overfit50.pkl',
        doscenes_anchor_only=True,
        doscenes_anchor_frames=None,
    ),
)

# === Optimizer: paramwise_cfg 凍 VAD,只訓 TextDeltaPlanner + LoRA ===
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            # FROZEN
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.0),
            # FROZEN: LLaVA backbone (base_model 含 embed_tokens, layers, etc.)
            # LoRA adapter weights 由 PEFT 用 .lora_A.* / .lora_B.* 命名 → 不在這條 match
            '_llava_model.base_model.model.': dict(lr_mult=0.0),
            '_llava_model.vision_tower': dict(lr_mult=0.0),
            '_llava_model.multi_modal_projector': dict(lr_mult=0.0),
            # TRAINABLE (lr_mult=1.0 by default — no need to list)
            # _text_delta_planner.text_proj, .mlp, .alpha → lr_mult=1.0
            # LoRA adapters '.lora_A.', '.lora_B.' → lr_mult=1.0
            '_text_delta_planner.alpha': dict(lr_mult=10.0),   # alpha 起點 0 需快
        }
    ),
    weight_decay=0.01,
)

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=1,    # single GPU 不需要 accum
    grad_clip=dict(max_norm=2, norm_type=2),
)

# === Schedule ===
total_epochs = 100
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=20, max_keep_ckpts=6)
evaluation = dict(interval=total_epochs)
log_config = dict(interval=10)
