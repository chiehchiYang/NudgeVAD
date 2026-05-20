"""Clean fine-tune v9c:VAD-only,*真* zero-language(cmd 從 lanelet 推)。

跟 v9 的差別:
  v9   pipeline:LoadDoScenesInstruction(override_cmd_from_text=True,
                                          regex on text → 3-class)
  v9c  pipeline:LoadLaneletCmd(用 sensor + nuScenes lanelet HD-map
                                 train 出來的 MLP 推 3-class)

所以 v9c 完全沒摸到 doScenes 文字 — model 看到的 cmd 只用了:
  * ego_his_trajs (8) + ego_lcf_feat (9) + can_bus (18) — 都 t≤0
  * 6 個 forward 投射點查 lanelet center (angle_diff, lat, presence)
  * 全部都是 *當前已知* 訊號,無 future leak、無 doScenes text

這給我們:
  1. cmd routing 在「真乾淨 zero-language」設定下的純基線
  2. 之後 stage-2 (加回 LLaVA) 可以看「語言對 ade_6s 的 incremental 貢獻」
     乾淨估計 — 因為 stage-1 的 cmd 不是從文字偷來的

Predictor 來源:
  tools/probe_cmd_predictor_lanelet.py train 出來的 81% accuracy MLP
  (val anchor 81.3%, train 91.4%)。已 dump 到 data/lanelet_cmd_predictor.pth。

預期 ade_6s:
  v6 baseline (cmd=[0,0,1]):       7.23 m   ← cmd 完全沒訊號
  v9c (cmd from lanelet, 81% acc): 中間值,大約 6.5-7.0 m
  v6 with-lang (cmd from text):    6.59 m   ← LLaVA + text 的上限
  ──────────
  v9c 跟 6.59 m 的差就是「語言對 cmd 的價值」。
"""
_base_ = ['./VAD_tiny_e2e_h4f12.py']

# Legacy custom_imports — not required for NudgeVAD; kept tolerant so the
# config still loads when the optional qwen3vl module isn't installed.
custom_imports = dict(
    imports=['qwen3vl'],
    allow_failed_imports=True,
)

vad_img_cache_dir_train = 'data/vad_img_cache/h4f12_train'
vad_img_cache_dir_val = 'data/vad_img_cache/h4f12_val'

# === 子定義 ===
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

# 注意:meta_keys 不再含 doScenes 相關 key(因為沒 LoadDoScenesInstruction)
clean_meta_keys = (
    'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img',
    'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor',
    'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
    'scene_token', 'can_bus', 'frame_idx',
    # 加 doScenes instruction-related meta keys 給 eval filter 用
    # (model 不會看,只是 eval_doscenes_local.py 過濾「有 instruction」的樣本)
    'ego_instruction', 'ego_instruction_type', 'ego_instruction_present',
    'has_static_reference', 'has_dynamic_reference',
)

# === Speed-class head ON,3-class cmd hard routing ===
model = dict(
    pts_bbox_head=dict(
        ego_fut_mode=3,
        enable_soft_cmd_routing=False,
        enable_speed_head=True,
        speed_bins=(0.5, 2.0, 5.0, 10.0, 15.0, 20.0),
        speed_dt=0.5,
        loss_speed_weight=0.1,
    ),
)

# === Lanelet cmd predictor 取代 LoadDoScenesInstruction ===
load_lanelet_cmd = dict(
    type='LoadLaneletCmd',
    predictor_path='data/lanelet_cmd_predictor.pth',
    dataroot='data/nuscenes',
)

# Eval 用:doScenes instruction string 進 img_metas(只給 eval 過濾,model 不讀),
# 不 overwrite cmd(cmd 由前面的 LoadLaneletCmd 已設定)。intent_match_check=True
# 跟 v6/v7/v8 eval 條件一致 → 同樣 62 個 anchor with-instruction 樣本,可公平比。
load_doscenes_inst_for_filter = dict(
    type='LoadDoScenesInstruction',
    ann_dir='third_party/doScenes/Annotations',
    scene_token_to_name_json='data/nuscenes/scene_token_to_name.json',
    mode='first',
    intent_match_check=True,
    override_cmd_from_text=False,
)

train_pipeline = [
    # Our `LoadVADImgCache` is a drop-in replacement for an internal qwen3vl
    # transform of the same name; it wraps `LoadMultiViewImageFromFiles` and
    # expands `img_shape` to per-view tuples as VAD's encoder expects.
    dict(type='LoadVADImgCache', cache_dir=vad_img_cache_dir_train),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    load_lanelet_cmd,
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(
        type='CustomCollect3D',
        meta_keys=clean_meta_keys,
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
    load_doscenes_inst_for_filter,
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
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
                 with_ego=True, with_label=False),
            dict(
                type='CustomCollect3D',
                meta_keys=clean_meta_keys,
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img',
                      'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                      'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                      'gt_attr_labels'],
            ),
        ],
    ),
]

# === Data:anchor-only,b=8 ===
batch_size = 8
data_root = 'data/nuscenes/'
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        doscenes_anchor_only=True,
    ),
    val=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        doscenes_anchor_only=True,
    ),
    test=dict(
        ann_file=data_root + 'vad_nuscenes_h4f12_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        doscenes_anchor_only=True,
    ),
)

total_epochs = 30
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)
evaluation = dict(interval=total_epochs)
log_config = dict(interval=20)
