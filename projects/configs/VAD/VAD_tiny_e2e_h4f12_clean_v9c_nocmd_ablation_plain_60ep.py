"""Stage 2: NudgeVAD adapter trained on Stage 1 ep60 frozen base.

Total compute matches Stage 1:
  Stage 1 (no-cmd baseline): 90 ep pure VAD trunk training
  Stage 2 (this):            60 ep trunk (= Stage 1 ep60 ckpt, frozen)
                           + 30 ep adapter training
                           = 90 ep equivalent

Key differences from earlier NudgeVAD trainings:
  - load_from = Stage 1 ep60 (instead of v4_resume/ep30)
  - cmd is randomized via ForceCmdNeutral(mode='random')
    so language adapter is the only stable direction signal
  - 30 ep (not 60), matching compute budget
"""
_base_ = ['./VAD_tiny_e2e_h4f12_clean_v9c_finetune_v4_resume_text_delta_scaleup.py']

# === New frozen base from Stage 1 ===
load_from = 'output_v9c_nocmd_anchor_only_90ep/epoch_60.pth'

# === Pipeline override: insert ForceCmdNeutral after LoadLaneletCmd ===
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
                   'sample_token',
                   'ego_instruction', 'ego_instruction_type', 'ego_instruction_present',
                   'has_static_reference', 'has_dynamic_reference')
vad_img_cache_dir_train = 'data/vad_img_cache/h4f12_train'

load_lanelet_cmd = dict(type='LoadLaneletCmd',
                        predictor_path='data/lanelet_cmd_predictor.pth',
                        dataroot='data/nuscenes')
force_random_cmd = dict(type='ForceCmdNeutral', mode='random')
load_doscenes_inst = dict(type='LoadDoScenesInstruction',
                          ann_dir='third_party/doScenes/Annotations',
                          scene_token_to_name_json='data/nuscenes/scene_token_to_name.json',
                          mode='random',
                          intent_match_check=False,
                          override_cmd_from_text=False)

train_pipeline = [
    dict(type='LoadVADImgCache', cache_dir=vad_img_cache_dir_train),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    load_lanelet_cmd,
    load_doscenes_inst,           # ← inject doScenes instruction
    force_random_cmd,             # ← override cmd to random one-hot (after instruction load)
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D', meta_keys=clean_meta_keys,
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
               'gt_attr_labels']),
]

data = dict(
    samples_per_gpu=2,            # avoid B=1 fancy-indexing bug
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
        doscenes_anchor_only=True,    # 700 sample/ep (frame_idx==4 only)
        doscenes_anchor_frames=None,
    ),
    # val/test pipelines come from _base_; we also need to inject ForceCmdNeutral there
    # but the eval at end of training is disabled (evaluation interval=total_epochs+1).
)

# === Adapter training: 60 ep (extended from 30 to ensure full convergence) ===
total_epochs = 60
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=10, max_keep_ckpts=7)
evaluation = dict(interval=total_epochs + 1)   # disable eval-during-train

optimizer = dict(lr=1e-4)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear', warmup_iters=100, warmup_ratio=1.0/3,
    min_lr_ratio=1e-3,
)
