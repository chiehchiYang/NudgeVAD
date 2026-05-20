"""v9c_finetune v4_resume + LLaVA branch (Phase 3).

Same compliance-safe training recipe as v4_resume (anchor +/- 1 window,
ramp 1->4, FDE 1.0, lanelet cmd via LoadLaneletCmd) but with the LLaVA
branch enabled on top:
  - model.type: VAD -> VADLLaVA (LLaVA-1.5-7b + LoRA + planning adapter)
  - train_pipeline: add LoadDoScenesInstruction *after* LoadLaneletCmd
    with override_cmd_from_text=False (lanelet cmd stays authoritative;
    LLaVA only sees the instruction text)
  - load_from: v4_resume epoch_30 (your 2.76 m anchor @6s ckpt)

ΔADE goal:
  Eval with --with-language and --no-language (both passes), then read
  off the per-horizon delta to isolate LLaVA's contribution under the
  compliance constraint.

Budget:
  4090 24 GB + LLaVA-1.5-7b fp16 + LoRA + grad-ckpt: ~19 GB at b=1.
  3 GPU x b=1 = effective batch 3 (vs v4_resume's effective 6 — slower
  but matches doc's "small batch is fine on small dataset" pattern).
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4_resume.py']

# === 從父鏈拿到的子定義 (避免 _base_ 變數泄漏) ===
fut_ts = 12
batch_size = 1
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
vad_img_cache_dir_train = 'data/vad_img_cache/h4f12_train'

# 加 doScenes instruction-related meta keys 給 model + eval filter 用
llava_meta_keys = (
    'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img',
    'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor',
    'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
    'scene_token', 'can_bus', 'frame_idx',
    # doScenes instruction (model reads via VADLLaVA._build_joint_drive_qa_text)
    'ego_instruction', 'ego_instruction_type', 'ego_instruction_present',
    'has_static_reference', 'has_dynamic_reference',
)

# === Lanelet cmd predictor (跟 v9c 同 — compliance-safe cmd) ===
load_lanelet_cmd = dict(
    type='LoadLaneletCmd',
    predictor_path='data/lanelet_cmd_predictor.pth',
    dataroot='data/nuscenes',
)

# === Train: 餵 instruction text 給 LLaVA,但不 overwrite cmd ===
load_doscenes_inst_train = dict(
    type='LoadDoScenesInstruction',
    ann_dir='third_party/doScenes/Annotations',
    scene_token_to_name_json='data/nuscenes/scene_token_to_name.json',
    mode='random',
    intent_match_check=False,   # 訓練不過濾 — 全部 anchor±1 樣本都要
    override_cmd_from_text=False,
)

# === Eval: mode='first' 跟 v9c eval 一致,intent_match_check=True 過濾 ===
load_doscenes_inst_eval = dict(
    type='LoadDoScenesInstruction',
    ann_dir='third_party/doScenes/Annotations',
    scene_token_to_name_json='data/nuscenes/scene_token_to_name.json',
    mode='first',
    intent_match_check=True,
    override_cmd_from_text=False,
)

train_pipeline = [
    dict(type='LoadVADImgCache', cache_dir=vad_img_cache_dir_train),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    load_lanelet_cmd,
    load_doscenes_inst_train,   # 加在 lanelet cmd 之後,文字進 ego_instruction
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(
        type='CustomCollect3D',
        meta_keys=llava_meta_keys,
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
              'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
              'gt_attr_labels'],
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    load_lanelet_cmd,
    load_doscenes_inst_eval,
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True, with_label=False),
            dict(
                type='CustomCollect3D',
                meta_keys=llava_meta_keys,
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img',
                      'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                      'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                      'gt_attr_labels'],
            ),
        ],
    ),
]

# === Load from v4_resume final ckpt (your 2.76 m anchor @6s) ===
load_from = 'output_v9c_finetune_v4_resume/epoch_30.pth'

# === Model: VAD -> VADLLaVA (keep v4's plan-head tricks via _base_ chain) ===
model = dict(
    type='VADLLaVA',
    drive_qa=False,
    llava_use_image=False,
    llava_enabled=True,
    llava_model_name='llava-hf/llava-1.5-7b-hf',
    llava_device='cuda',
    llava_dtype='float16',
    llava_replace_ego_fut_preds=False,
    llava_freeze=True,
    llava_use_planning_adapter=True,
    llava_use_projector=False,
    llava_adapter_query_tokens=8,
    llava_adapter_num_layers=2,
    llava_adapter_num_heads=8,
    llava_adapter_dropout=0.0,
    llava_adapter_internal_dim=768,
    llava_use_lora=True,
    llava_hidden_size_hint=4096,
    llava_retry_init=False,
    llava_gradient_checkpointing=True,   # 4090 24 GB — keep ckpt ON
    llava_checkpoint_mode='no_llava_base',
    llava_use_plan_constraint_loss=True,
    llava_plan_loss_weight=1.0,
    llava_qa_loss_weight=0.0,
    llava_text_max_length=384,
    llava_image_pool_stride=2,
    llava_lora_r=16,
    llava_lora_alpha=32,
    llava_lora_dropout=0.05,
    llava_lora_target_modules=['q_proj', 'v_proj'],
)

# === Data: override pipelines + samples_per_gpu ===
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,   # LLaVA worker spawn 較重,降 worker
    train=dict(
        pipeline=train_pipeline,
        fut_ts=fut_ts,
        drive_qa=False,
    ),
    val=dict(
        pipeline=test_pipeline,
        fut_ts=fut_ts,
        drive_qa=False,
    ),
    test=dict(
        pipeline=test_pipeline,
        fut_ts=fut_ts,
        drive_qa=False,
    ),
)

# === Short LLaVA add-on finetune ===
# Base 已收斂 (v4_resume @6s anchor 2.76 m),只訓 LLaVA branch + LoRA + adapter,
# 10 ep 應足夠看到 ΔADE 趨勢。LR 沿用 v4_resume 的 1e-4 cosine。
total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=2, max_keep_ckpts=total_epochs // 2 + 1)
evaluation = dict(interval=total_epochs)
log_config = dict(interval=20)
