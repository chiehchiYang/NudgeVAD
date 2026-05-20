import math
import re
import numpy as np
import torch
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)


@PIPELINES.register_module()
class CustomObjectRangeFilter(object):
    """Filter objects by the range, and also filter corresponding fut trajs

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_attr_labels = input_dict['attr_labels']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]
        gt_attr_labels = gt_attr_labels[mask.numpy().astype(bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['gt_attr_labels'] = gt_attr_labels

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class CustomObjectNameFilter(object):
    """Filter GT objects by their names, , and also filter corresponding fut trajs

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        input_dict['gt_attr_labels'] = input_dict['gt_attr_labels'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        results['img_shape_before_pad'] = [img.shape for img in results['img']]
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str



@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 optional_keys=(),
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus',
                            )):
        self.keys = keys
        self.optional_keys = set(optional_keys)
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
      
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            if key in results:
                data[key] = results[key]
            elif key not in self.optional_keys:
                raise KeyError(f'{key} is not found in results')
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'



@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        results['img_shape_before_scale'] = [img.shape for img in results['img']]
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class PrepareDriveQAMosaic(object):
    def __init__(
        self,
        enabled=True,
        include_image=True,
        tile_size=336,
        marker_radius=4,
        marker_color=(255, 64, 64),
        sample_categories=('Perception', 'Prediction', 'Planning'),
        sample_num_per_category=1,
        camera_order=(
            'CAM_FRONT_LEFT',
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT',
            'CAM_BACK',
            'CAM_BACK_RIGHT',
        ),
    ):
        self.enabled = enabled
        self.include_image = include_image
        self.tile_size = int(tile_size)
        self.marker_radius = int(marker_radius)
        self.marker_color = np.array(marker_color, dtype=np.uint8)
        if isinstance(sample_categories, str):
            sample_categories = (sample_categories,)
        self.sample_categories = tuple(sample_categories)
        self.sample_num_per_category = int(sample_num_per_category)
        if self.sample_num_per_category < 0:
            raise ValueError('sample_num_per_category must be >= 0.')
        self.camera_order = tuple(camera_order)
        self._coord_pattern = re.compile(
            r'<\s*([^,>]+)\s*,\s*(CAM_[A-Z_]+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*>'
        )

    def _denormalize_img(self, img, img_norm_cfg):
        if img_norm_cfg is None:
            return np.clip(img, 0, 255).astype(np.uint8)
        mean = np.array(img_norm_cfg.get('mean', [0.0, 0.0, 0.0]), dtype=np.float32)
        std = np.array(img_norm_cfg.get('std', [1.0, 1.0, 1.0]), dtype=np.float32)
        denorm = img.astype(np.float32) * std + mean
        return np.clip(denorm, 0, 255).astype(np.uint8)

    def _tile_position(self, camera_name):
        camera_idx = self.camera_order.index(camera_name)
        row = camera_idx // 3
        col = camera_idx % 3
        return row, col

    def _build_camera_to_index(self, camera_names):
        return {name: idx for idx, name in enumerate(camera_names)}

    def _transform_coordinate(self, camera_name, x, y, camera_to_index, raw_shapes, scaled_shapes, pad_shapes):
        if camera_name not in camera_to_index or camera_name not in self.camera_order:
            return x, y, None

        cam_idx = camera_to_index[camera_name]
        raw_h, raw_w = raw_shapes[cam_idx][:2]
        scaled_h, scaled_w = scaled_shapes[cam_idx][:2]
        pad_h, pad_w = pad_shapes[cam_idx][:2]
        if raw_w <= 0 or raw_h <= 0 or scaled_w <= 0 or scaled_h <= 0 or pad_w <= 0 or pad_h <= 0:
            return x, y, None

        scale_x = float(scaled_w) / float(raw_w)
        scale_y = float(scaled_h) / float(raw_h)
        x_proc = float(x) * scale_x
        y_proc = float(y) * scale_y

        row, col = self._tile_position(camera_name)
        x_tile = x_proc / float(pad_w) * float(self.tile_size)
        y_tile = y_proc / float(pad_h) * float(self.tile_size)
        x_mosaic = x_tile + col * self.tile_size
        y_mosaic = y_tile + row * self.tile_size
        return x_mosaic, y_mosaic, (x_mosaic, y_mosaic)

    def _transform_text(self, text, camera_to_index, raw_shapes, scaled_shapes, pad_shapes, collect_points=False):
        points = []
        if not isinstance(text, str):
            return text, points

        def replace(match):
            obj_id, camera_name, x_str, y_str = match.groups()
            x_mosaic, y_mosaic, point = self._transform_coordinate(
                camera_name,
                float(x_str),
                float(y_str),
                camera_to_index,
                raw_shapes,
                scaled_shapes,
                pad_shapes,
            )
            if collect_points and point is not None:
                points.append(point)
            return f'<{obj_id},{camera_name},{x_mosaic:.1f},{y_mosaic:.1f}>'

        return self._coord_pattern.sub(replace, text), points

    def _draw_points(self, mosaic, points):
        for x, y in points:
            xi = int(round(x))
            yi = int(round(y))
            x0 = max(0, xi - self.marker_radius)
            x1 = min(mosaic.shape[1], xi + self.marker_radius + 1)
            y0 = max(0, yi - self.marker_radius)
            y1 = min(mosaic.shape[0], yi + self.marker_radius + 1)
            mosaic[y0:y1, x0:x1] = self.marker_color
        return mosaic

    def _sample_drive_qa(self, drive_qa):
        samples = []
        if not isinstance(drive_qa, dict) or self.sample_num_per_category == 0:
            return samples
        for category in self.sample_categories:
            qa_group = drive_qa.get(category, None)
            if not isinstance(qa_group, list) or len(qa_group) == 0:
                continue
            valid_qas = []
            for qa_item in qa_group:
                if not isinstance(qa_item, dict):
                    continue
                question = qa_item.get('Q')
                answer = qa_item.get('A')
                if not isinstance(question, str) or not isinstance(answer, str):
                    continue
                valid_qas.append(dict(question=question, answer=answer))
            if len(valid_qas) == 0:
                continue

            sample_count = min(self.sample_num_per_category, len(valid_qas))
            sampled_indices = np.atleast_1d(
                random.choice(len(valid_qas), size=sample_count, replace=False)
            ).tolist()
            for qa_idx in sampled_indices:
                qa_item = valid_qas[int(qa_idx)]
                samples.append(
                    dict(
                        category=category,
                        question=qa_item['question'],
                        answer=qa_item['answer'],
                    )
                )
        return samples

    def __call__(self, results):
        if not self.enabled:
            return results

        drive_qa = results.get('drive_qa', None)
        sampled_qas = self._sample_drive_qa(drive_qa)
        transformed_samples = [
            dict(
                category=sample['category'],
                question=sample['question'],
                answer=sample['answer'],
                original_question=sample['question'],
                original_answer=sample['answer'],
            )
            for sample in sampled_qas
        ]

        if len(transformed_samples) > 0:
            results['drive_qa_samples'] = transformed_samples

        if not self.include_image:
            return results

        camera_names = results.get('camera_names', None)
        imgs = results.get('img', None)
        if camera_names is None or imgs is None:
            return results

        camera_to_index = self._build_camera_to_index(camera_names)
        raw_shapes = results.get('img_shape_before_scale', None)
        if raw_shapes is None:
            raw_shapes = [img.shape for img in imgs]
        scaled_shapes = results.get('ori_shape', None)
        if scaled_shapes is None:
            scaled_shapes = [img.shape for img in imgs]
        pad_shapes = results.get('pad_shape', None)
        if pad_shapes is None:
            pad_shapes = [img.shape for img in imgs]

        img_norm_cfg = results.get('img_norm_cfg', None)
        ordered_tiles = []
        for camera_name in self.camera_order:
            if camera_name not in camera_to_index:
                ordered_tiles.append(np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8))
                continue
            cam_idx = camera_to_index[camera_name]
            tile = self._denormalize_img(imgs[cam_idx], img_norm_cfg)
            tile = mmcv.imresize(tile, (self.tile_size, self.tile_size), return_scale=False)
            ordered_tiles.append(tile.astype(np.uint8))

        mosaic_rows = []
        for row_idx in range(0, len(ordered_tiles), 3):
            mosaic_rows.append(np.concatenate(ordered_tiles[row_idx:row_idx + 3], axis=1))
        mosaic = np.concatenate(mosaic_rows, axis=0)

        transformed_samples = []
        marker_points = []
        for sample in sampled_qas:
            question_t, points = self._transform_text(
                sample['question'],
                camera_to_index,
                raw_shapes,
                scaled_shapes,
                pad_shapes,
                collect_points=True,
            )
            answer_t, _ = self._transform_text(
                sample['answer'],
                camera_to_index,
                raw_shapes,
                scaled_shapes,
                pad_shapes,
                collect_points=False,
            )
            marker_points.extend(points)
            transformed_samples.append(
                dict(
                    category=sample['category'],
                    question=question_t,
                    answer=answer_t,
                    original_question=sample['question'],
                    original_answer=sample['answer'],
                )
            )

        mosaic = self._draw_points(mosaic, marker_points)
        results['llava_mosaic_img'] = mosaic
        if len(transformed_samples) > 0:
            results['drive_qa_samples'] = transformed_samples
        return results


@PIPELINES.register_module()
class LoadDoScenesInstruction(object):
    """Inject one doScenes natural-language instruction into the pipeline results.

    The transform builds two indices once at construction:
      * scene_token -> scene_name (from a precomputed JSON map)
      * scene_name -> list[(instruction, instruction_type)] (from doScenes CSVs)
    At call time it looks up the current sample's scene by
    ``results['scene_token']`` and writes the chosen instruction to
    ``results[instruction_key]`` (default ``ego_instruction``).

    Args:
        ann_dir (str): Path to the doScenes Annotations directory containing
            CSV files with columns ``Scene Number``, ``Instruction Type``,
            ``Instruction``.
        scene_token_to_name_json (str): Path to a JSON file mapping each
            nuScenes scene token to its scene name (e.g. ``scene-0036``).
        mode (str): ``'random'`` picks one instruction per call (training);
            ``'first'`` picks instructions deterministically (val/test).
        instruction_key (str): Result key for the instruction string.
        skip_if_missing (bool): When True, scenes without doScenes annotation
            yield an empty instruction; when False the call raises.
    """

    def __init__(self,
                 ann_dir,
                 scene_token_to_name_json,
                 mode='random',
                 instruction_key='ego_instruction',
                 skip_if_missing=True,
                 intent_match_check=False,
                 override_cmd_from_text=False,
                 cmd_num_classes=3):
        import csv
        import glob
        import json
        import os

        if mode not in ('random', 'first'):
            raise ValueError(f'mode must be random|first, got {mode!r}')
        self.mode = mode
        self.instruction_key = instruction_key
        self.skip_if_missing = bool(skip_if_missing)
        # When True, drop candidate instructions whose keyword-derived intent
        # disagrees with the GT trajectory's intent (computed from the future
        # offsets). If no candidate matches, the sample is treated as if no
        # instruction were available (instruction='', present=False) so the
        # model can still train on the perception aux losses but the LLaVA
        # branch isn't fed a misleading (instruction, GT) pair.
        self.intent_match_check = bool(intent_match_check)
        # When True, derive ego_fut_cmd from the picked instruction text
        # (regex → 3-class one-hot) and overwrite results['ego_fut_cmd'],
        # so the model never sees the GT-future-derived cmd. Required for
        # doScenes "no future leakage" compliance. When no instruction is
        # available, defaults to [0, 0, 1] (Go Straight).
        self.override_cmd_from_text = bool(override_cmd_from_text)
        # Number of cmd classes the override emits.
        #   3 → VAD default [Turn Right, Turn Left, Go Straight] (v6)
        #   6 → finer-grained [turn_right, turn_left, stop, slow_yield,
        #                      change_lane (∪pass_overtake),
        #                      forward (∪follow ∪other)] (v8)
        # Model's `ego_fut_mode` must match this.
        if int(cmd_num_classes) not in (3, 6):
            raise ValueError(f'cmd_num_classes must be 3 or 6, got {cmd_num_classes}')
        self.cmd_num_classes = int(cmd_num_classes)

        with open(scene_token_to_name_json, 'r') as f:
            self._token_to_name = json.load(f)

        self._scene_to_inst = {}
        files = sorted(glob.glob(os.path.join(ann_dir, '*.csv')))
        if not files:
            raise FileNotFoundError(
                f'No doScenes CSVs found under {ann_dir}')
        for path in files:
            with open(path, newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    raw = (row.get('Scene Number') or '').strip()
                    inst = (row.get('Instruction') or '').strip()
                    itype = (row.get('Instruction Type') or '').strip()
                    if not raw or not inst:
                        continue
                    try:
                        scene_num = int(float(raw))
                    except ValueError:
                        continue
                    name = f'scene-{scene_num:04d}'
                    self._scene_to_inst.setdefault(name, []).append(
                        (inst, itype))

    def _decode_type(self, code):
        c = (code or '').strip().lower()
        return {
            'has_static_reference': 's' in c,
            'has_dynamic_reference': 'd' in c,
        }

    # ---- Intent matching helpers ----------------------------------------
    # Order matters; first match wins. Mirrors `tools/build_clean_doscenes_pkl`.
    _TEXT_INTENT_PATTERNS = None  # lazy-built regex list

    @classmethod
    def _ensure_intent_patterns(cls):
        if cls._TEXT_INTENT_PATTERNS is not None:
            return
        import re as _re
        cls._TEXT_INTENT_PATTERNS = [
            ('stop',          _re.compile(r'\b(stop|halt)\b', _re.I)),
            ('slow_yield',    _re.compile(r'\b(slow|yield|brake|wait|pause)\b', _re.I)),
            ('turn_left',     _re.compile(r'\bturn\s+left\b|\bleft\s+turn\b|\bmake\s+a\s+left\b', _re.I)),
            ('turn_right',    _re.compile(r'\bturn\s+right\b|\bright\s+turn\b|\bmake\s+a\s+right\b', _re.I)),
            ('change_lane',   _re.compile(r'\b(shift|change|switch|merge).{0,20}\blane\b', _re.I)),
            ('follow',        _re.compile(r'\b(follow|behind)\b', _re.I)),
            ('pass_overtake', _re.compile(r'\b(pass|overtake)\b', _re.I)),
            ('forward',       _re.compile(r'\b(go|continue|move|drive|straight|forward)\b', _re.I)),
        ]

    @classmethod
    def _text_intent(cls, inst):
        cls._ensure_intent_patterns()
        for tag, rx in cls._TEXT_INTENT_PATTERNS:
            if rx.search(inst or ''):
                return tag
        return 'other'

    @staticmethod
    def _gt_intent(gt_offsets):
        """Coarse intent from the GT 12-step future trajectory.

        Coordinate frame is LiDAR-local (X right, Y forward); ±2 m matches
        the converter's `ego_fut_cmd` rule. Stop is detected as low speed
        on the final 4 steps (= 2 s).
        """
        import numpy as _np
        gt = _np.asarray(gt_offsets, dtype=_np.float64).reshape(-1, 2)
        cum = _np.cumsum(gt, axis=0)
        final_lat = float(cum[-1, 0])
        final_fwd = float(cum[-1, 1])
        speed_last = float(_np.linalg.norm(gt[-4:].mean(0)))
        if final_lat <= -2:
            return 'turn_left'
        if final_lat >= 2:
            return 'turn_right'
        if speed_last < 0.5 and final_fwd < 5.0:
            return 'stop'
        return 'forward'

    _FORWARD_FAMILY = frozenset({
        'forward', 'follow', 'pass_overtake', 'change_lane', 'slow_yield', 'other'
    })

    @staticmethod
    def _intent_to_cmd(text_intent, num_classes=3):
        """Map 8-class text intent → cmd one-hot.

        num_classes=3 (v6 default, matches converter order):
          [Turn Right, Turn Left, Go Straight]
          — speed nuance (stop / slow / follow) collapses to Straight,
            captured separately by the v5 speed-class head.

        num_classes=6 (v8 finer-grained, matches doScenes intent
        distribution; each class ≥ 6% of samples):
          [turn_right, turn_left, stop, slow_yield,
           change_lane (∪pass_overtake), forward (∪follow ∪other)]
        """
        if num_classes == 3:
            if text_intent == 'turn_right':
                return np.array([1, 0, 0], dtype=np.float32)
            if text_intent == 'turn_left':
                return np.array([0, 1, 0], dtype=np.float32)
            return np.array([0, 0, 1], dtype=np.float32)
        if num_classes == 6:
            out = np.zeros(6, dtype=np.float32)
            if text_intent == 'turn_right':
                out[0] = 1.0
            elif text_intent == 'turn_left':
                out[1] = 1.0
            elif text_intent == 'stop':
                out[2] = 1.0
            elif text_intent == 'slow_yield':
                out[3] = 1.0
            elif text_intent in ('change_lane', 'pass_overtake'):
                out[4] = 1.0
            else:                                   # forward / follow / other
                out[5] = 1.0
            return out
        raise ValueError(f'num_classes must be 3 or 6, got {num_classes}')

    @classmethod
    def _aligned(cls, text_t, gt_t):
        if text_t == gt_t:
            return True
        if text_t in cls._FORWARD_FAMILY and gt_t == 'forward':
            return True
        return False

    def __call__(self, results):
        scene_token = results.get('scene_token')
        scene_name = self._token_to_name.get(scene_token) if scene_token else None
        instructions = self._scene_to_inst.get(scene_name, []) if scene_name else []

        # Optionally drop candidates whose intent disagrees with the GT.
        if self.intent_match_check and instructions:
            gt_offsets = results.get('ego_fut_trajs')
            if gt_offsets is not None:
                gt_t = self._gt_intent(gt_offsets)
                instructions = [
                    (inst, itype) for (inst, itype) in instructions
                    if self._aligned(self._text_intent(inst), gt_t)
                ]

        if not instructions:
            if not self.skip_if_missing:
                raise KeyError(
                    f'No doScenes instruction for scene_token={scene_token!r} '
                    f'(scene_name={scene_name!r}).')
            results[self.instruction_key] = ''
            results[f'{self.instruction_key}_type'] = ''
            results[f'{self.instruction_key}_present'] = False
            results['has_static_reference'] = False
            results['has_dynamic_reference'] = False
            if self.override_cmd_from_text:
                results['ego_fut_cmd'] = self._intent_to_cmd(
                    'other', num_classes=self.cmd_num_classes)
            return results

        if self.mode == 'random':
            idx = int(random.randint(0, len(instructions)))
        else:
            idx = 0
        inst, itype = instructions[idx]
        results[self.instruction_key] = inst
        results[f'{self.instruction_key}_type'] = itype
        results[f'{self.instruction_key}_present'] = True
        results.update(self._decode_type(itype))
        if self.override_cmd_from_text:
            results['ego_fut_cmd'] = self._intent_to_cmd(
                self._text_intent(inst), num_classes=self.cmd_num_classes)
        return results


@PIPELINES.register_module()
class LoadLaneletCmd:
    """Predict ego_fut_cmd from sensor + nuScenes lanelet HD-map only —
    *no language*, *no GT future*. Pure compliance-clean cmd source.

    Loads a small MLP predictor (saved by tools/probe_cmd_predictor_lanelet.py)
    that maps a 53-d feature
        [ego_his_trajs(8) + ego_lcf_feat(9) + can_bus(18)
         + lanelet_feat(18 = 6 forward query × 3 dims)]
    → 3-class one-hot cmd.

    Lanelet feature: at construction time we discretise every lane and
    lane_connector polyline in the 4 nuScenes maps and build per-map KDTrees.
    At call time we project the ego forward at d ∈ {5, 10, 15, 20, 25, 30} m
    along its current global yaw, find the nearest lane-center point within
    a 6-m radius, and emit (angle_diff, lateral_offset, presence).

    Per-sample cost ≈ 5 ms (KDTree queries + tiny MLP forward); init cost
    ≈ 8 s (map loading + lane discretisation).

    This transform OVERWRITES results['ego_fut_cmd']. Place it after
    LoadAnnotations3D (so input_dict already has 'ego2global_translation',
    'ego2global_rotation', 'map_location', 'can_bus' from get_data_info).

    Args:
        predictor_path (str): Path to .pth dumped by probe_cmd_predictor_lanelet.
        dataroot (str): nuScenes dataroot for NuScenesMap loading.
    """

    # Class-level cache so multiple instances (e.g. train + test pipeline copies)
    # share the heavy NuScenesMap + KDTree state.
    _LANELET_CACHE = None
    _MAP_LOCS = ('singapore-onenorth', 'singapore-hollandvillage',
                 'singapore-queenstown', 'boston-seaport')

    def __init__(self,
                 predictor_path='data/lanelet_cmd_predictor.pth',
                 dataroot='data/nuscenes'):
        # Keep dataroot picklable for lazy cache rebuild in spawn-mode workers
        self._dataroot = dataroot

        ckpt = torch.load(predictor_path, map_location='cpu', weights_only=False)
        self._mean = ckpt['mean']                       # (1, in_dim)
        self._std = ckpt['std']
        self._n_class = int(ckpt['n_class'])
        self._query_ds = tuple(ckpt['lanelet_query_ds'])
        self._max_radius = float(ckpt['lanelet_max_radius'])
        self._resolution = float(ckpt['lanelet_resolution'])

        # tiny MLP — same architecture as probe's CmdMLP
        in_dim = int(ckpt['in_dim'])
        hidden = int(ckpt['hidden'])
        n_class = self._n_class
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_class),
        )
        # Probe's CmdMLP wraps Sequential in self.net → strip 'net.' prefix
        sd = {k[len('net.'):]: v for k, v in ckpt['state_dict'].items()
              if k.startswith('net.')}
        self._mlp.load_state_dict(sd)
        self._mlp.eval()

        # In main process: build cache eagerly so first __call__ is fast.
        # In spawn workers (where __init__ doesn't re-run after unpickle),
        # _lanelet_feat below also lazy-builds if cache is None.
        self._build_map_cache(dataroot)

    @classmethod
    def _build_map_cache(cls, dataroot):
        if cls._LANELET_CACHE is not None:
            return
        from scipy.spatial import cKDTree
        from nuscenes.map_expansion.map_api import NuScenesMap
        from nuscenes.map_expansion import arcline_path_utils

        cache = {}
        for loc in cls._MAP_LOCS:
            nmap = NuScenesMap(dataroot=dataroot, map_name=loc)
            pts = []
            tokens = [r['token'] for r in nmap.lane] + \
                     [r['token'] for r in nmap.lane_connector]
            for tok in tokens:
                try:
                    arc = nmap.get_arcline_path(tok)
                except KeyError:
                    continue
                disc = arcline_path_utils.discretize_lane(arc, resolution_meters=1.0)
                for x, y, yaw in disc:
                    pts.append((x, y, yaw))
            arr = np.asarray(pts, dtype=np.float64)
            tree = cKDTree(arr[:, :2])
            cache[loc] = (arr, tree)
        cls._LANELET_CACHE = cache

    @staticmethod
    def _ego_yaw_global(rotation_quat):
        from pyquaternion import Quaternion
        q = Quaternion(rotation_quat)
        fwd = q.rotate([1.0, 0.0, 0.0])
        return float(math.atan2(fwd[1], fwd[0]))

    def _lanelet_feat(self, ego_xy, ego_yaw, map_loc):
        if LoadLaneletCmd._LANELET_CACHE is None:
            # Spawn worker received unpickled instance — rebuild cache here
            LoadLaneletCmd._build_map_cache(self._dataroot)
        cache = LoadLaneletCmd._LANELET_CACHE
        if map_loc not in cache:
            return np.zeros(len(self._query_ds) * 3, dtype=np.float32)
        arr, tree = cache[map_loc]
        out = np.zeros(len(self._query_ds) * 3, dtype=np.float32)
        cy, sy = math.cos(ego_yaw), math.sin(ego_yaw)
        for i, d in enumerate(self._query_ds):
            qx = ego_xy[0] + d * cy
            qy = ego_xy[1] + d * sy
            dist, idx = tree.query([qx, qy], distance_upper_bound=self._max_radius)
            if not np.isfinite(dist) or dist >= self._max_radius:
                continue
            lane_xy = arr[idx, :2]
            lane_yaw = float(arr[idx, 2])
            angle_diff = (lane_yaw - ego_yaw + math.pi) % (2 * math.pi) - math.pi
            dx = lane_xy[0] - qx
            dy = lane_xy[1] - qy
            lat = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy
            out[3 * i + 0] = angle_diff
            out[3 * i + 1] = lat
            out[3 * i + 2] = 1.0
        return out

    def __call__(self, results):
        # Sensor 35-d
        his = np.asarray(results['ego_his_trajs'], dtype=np.float32).reshape(-1)
        lcf = np.asarray(results['ego_lcf_feat'], dtype=np.float32).reshape(-1)
        cb  = np.asarray(results.get('can_bus', np.zeros(18)), dtype=np.float32).reshape(-1)
        cb  = cb[:18] if cb.size >= 18 else np.pad(cb, (0, 18 - cb.size))
        sensor = np.concatenate([his, lcf, cb], axis=0)
        # Lanelet 18-d
        ego_xy  = results['ego2global_translation'][:2]
        ego_yaw = self._ego_yaw_global(results['ego2global_rotation'])
        ll = self._lanelet_feat(ego_xy, ego_yaw, results.get('map_location'))
        feat = np.concatenate([sensor, ll], axis=0).astype(np.float32)
        # Standardise + MLP forward
        feat_n = (feat[None, :] - self._mean) / (self._std + 1e-12)
        with torch.no_grad():
            logits = self._mlp(torch.from_numpy(feat_n.astype(np.float32)))
            cls = int(logits.argmax(dim=-1).item())
        # Write one-hot into results
        cmd = np.zeros(self._n_class, dtype=np.float32)
        cmd[cls] = 1.0
        results['ego_fut_cmd'] = cmd
        return results


@PIPELINES.register_module()
class ForceCmdNeutral(object):
    """Override `ego_fut_cmd` to a fixed or random value.

    Caveat: VAD_head's plan_bound/plan_col losses index ego_fut_preds with
    `ego_fut_cmd == 1` (binary one-hot mask). A constant non-one-hot value
    like [1/3,1/3,1/3] makes the mask empty → RuntimeError. Use mode='random'
    to randomly pick a one-hot cmd at every sample so the mask is well-formed
    but the cmd carries no useful signal.

    Args:
        mode (str): 'random' picks a random one-hot at each sample;
                    'constant' uses `value` directly (works only if `value`
                    is a valid one-hot).
        value (tuple[float]): cmd values when mode='constant'.
    """
    def __init__(self, mode='random', value=(0.0, 0.0, 1.0)):
        if mode not in ('random', 'constant'):
            raise ValueError(f'mode must be random|constant, got {mode}')
        self.mode = mode
        self.value = np.array(value, dtype=np.float32)

    def __call__(self, results):
        if self.mode == 'random':
            cmd = np.zeros(3, dtype=np.float32)
            cmd[np.random.randint(0, 3)] = 1.0
            results['ego_fut_cmd'] = cmd
        else:
            results['ego_fut_cmd'] = self.value.copy()
        return results

    def __repr__(self):
        if self.mode == 'random':
            return 'ForceCmdNeutral(mode=random)'
        return f'ForceCmdNeutral(mode=constant, value={self.value.tolist()})'


@PIPELINES.register_module()
class CustomPointsRangeFilter:
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        data["points"] = clean_points
        return data
