"""doScenes h4f12 baseline config (no LLaVA).

Inherits VAD_tiny_e2e.py and overrides:
  * dataset ann_file -> vad_nuscenes_h4f12_*.pkl (4 history + 12 future timesteps)
  * head fut_ts/valid_fut_ts -> 12
  * pipeline -> insert LoadDoScenesInstruction so each sample carries
    `ego_instruction` / `ego_instruction_type` in img_metas
"""
_base_ = ['./VAD_tiny_e2e.py']

# --- doScenes spec parameters ---
fut_ts = 12
his_ts = 4

# --- Pipeline: replicate parent's, plus LoadDoScenesInstruction ---
data_root = 'data/nuscenes/'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Inject doScenes instruction *after* LoadAnnotations so we have scene_token
# in results.  We register it in CustomCollect3D.meta_keys so the strings flow
# through DataContainer into img_metas without breaking tensor batching.
load_doscenes_instruction_train = dict(
    type='LoadDoScenesInstruction',
    ann_dir='third_party/doScenes/Annotations',
    scene_token_to_name_json='data/nuscenes/scene_token_to_name.json',
    mode='random',
)
load_doscenes_instruction_test = dict(
    type='LoadDoScenesInstruction',
    ann_dir='third_party/doScenes/Annotations',
    scene_token_to_name_json='data/nuscenes/scene_token_to_name.json',
    mode='first',
)

doscenes_meta_keys = (
    'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img',
    'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor',
    'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
    'scene_token', 'can_bus',
    # frame_idx exposes the keyframe position within its scene (anchor = 4).
    # Needed by eval / demo to gate to the doScenes anchor-only protocol.
    'frame_idx',
    # doScenes additions:
    'ego_instruction', 'ego_instruction_type', 'ego_instruction_present',
    'has_static_reference', 'has_dynamic_reference',
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    load_doscenes_instruction_train,
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(
        type='CustomCollect3D',
        meta_keys=doscenes_meta_keys,
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
              'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
              'gt_attr_labels'],
    ),
]

file_client_args = dict(backend='disk')
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    load_doscenes_instruction_test,
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
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            dict(
                type='CustomCollect3D',
                meta_keys=doscenes_meta_keys,
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img',
                      'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                      'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                      'gt_attr_labels'],
            ),
        ],
    ),
]

# --- Model: bump fut_ts in pts_bbox_head; loss / decoder layers re-shape automatically ---
model = dict(
    pts_bbox_head=dict(
        fut_ts=fut_ts,
        valid_fut_ts=fut_ts,
    ),
)

# --- Data: point at h4f12 pkls and pass fut_ts down to dataset ---
data = dict(
    train=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_train.pkl',
        fut_ts=fut_ts,
        pipeline=train_pipeline,
    ),
    val=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_val.pkl',
        fut_ts=fut_ts,
        pipeline=test_pipeline,
    ),
    test=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_val.pkl',
        fut_ts=fut_ts,
        pipeline=test_pipeline,
    ),
)
