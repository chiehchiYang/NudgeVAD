"""Stage 1: Pure VAD trained with NO cmd signal (cmd → neutral [1/3,1/3,1/3]).

Setup:
  - load_from: ckpts/shu_wei_stripped_for_v9c.pth (clean from base)
  - 90 epochs on anchor-only (700 samples/ep, frame_idx == 4 only)
  - cmd channel forced to neutral via ForceCmdNeutral transform
    (placed AFTER LoadLaneletCmd → overrides whatever lanelet predicts)
  - Pure VAD (no LLaVA, no LoRA, no TextDeltaPlanner)

Compute budget:
  - 8-GPU DDP, samples_per_gpu=1, accum=1 → eff_batch 8
  - ~1-2 hr wall-clock (700×90/8/8 ≈ 985 iter total)

Purpose: provide compute-fair baseline + frozen base for Stage 2 NudgeVAD
  adapter. Stage 2 loads from this ep60 ckpt, freezes trunk, trains adapter
  30 ep — equal total compute (90 ep) vs continuing this baseline to ep90.
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c.py']

load_from = 'ckpts/shu_wei_stripped_for_v9c.pth'

# === Pipeline: insert ForceCmdNeutral after LoadLaneletCmd ===
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
clean_meta_keys = ('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
                   'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
                   'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                   'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor',
                   'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'scene_token',
                   'can_bus', 'fut_valid_flag', 'lidar2ego', 'lidar2global', 'frame_idx',
                   'camera_names', 'lidar2cam', 'camera2ego', 'camera_intrinsics',
                   'sample_token')
vad_img_cache_dir_train = 'data/vad_img_cache/h4f12_train'
vad_img_cache_dir_val = 'data/vad_img_cache/h4f12_val'

load_lanelet_cmd = dict(type='LoadLaneletCmd',
                        predictor_path='data/lanelet_cmd_predictor.pth',
                        dataroot='data/nuscenes')
force_neutral_cmd = dict(type='ForceCmdNeutral', mode='random')

train_pipeline = [
    dict(type='LoadVADImgCache', cache_dir=vad_img_cache_dir_train),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    load_lanelet_cmd,
    force_neutral_cmd,            # ← override cmd to neutral
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D', meta_keys=clean_meta_keys,
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
               'gt_attr_labels']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    load_lanelet_cmd,
    force_neutral_cmd,            # ← override cmd to neutral at eval too
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1600, 900), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
             dict(type='PadMultiViewImage', size_divisor=32),
             dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
                  with_ego=True, with_label=False),
             dict(type='CustomCollect3D', meta_keys=clean_meta_keys,
                  keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img',
                        'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                        'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                        'gt_attr_labels']),
         ]),
]

data = dict(
    samples_per_gpu=2,            # B=1 triggers fancy-indexing bug in plan_loss
    workers_per_gpu=4,            # half of default to save RAM
    train=dict(
        pipeline=train_pipeline,
        doscenes_anchor_only=True,    # 700 samples/ep, frame_idx==4 only
        doscenes_anchor_frames=None,
    ),
    val=dict(pipeline=test_pipeline, doscenes_anchor_only=True),
    test=dict(pipeline=test_pipeline, doscenes_anchor_only=True),
)

# === Training schedule: 90 ep cosine ===
total_epochs = 90
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=10, max_keep_ckpts=10)
evaluation = dict(interval=total_epochs)   # only eval at end

optimizer = dict(lr=1e-4)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear', warmup_iters=100, warmup_ratio=1.0/3,
    min_lr_ratio=1e-3,
)
