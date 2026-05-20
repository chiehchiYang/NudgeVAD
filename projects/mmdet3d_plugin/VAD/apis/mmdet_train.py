import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook
from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset


@HOOKS.register_module()
class Fp16OptimizerHookWithGradCheck(Fp16OptimizerHook):
    """Fp16 optimizer hook with first non-finite gradient diagnostics."""

    @staticmethod
    def _find_first_nonfinite_grad(model):
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            finite_mask = torch.isfinite(grad)
            if finite_mask.all():
                continue
            total = int(grad.numel())
            finite_num = int(finite_mask.sum().item())
            nonfinite_num = total - finite_num
            finite_ratio = float(finite_num / max(total, 1))
            return {
                "name": name,
                "dtype": str(grad.dtype),
                "shape": tuple(grad.shape),
                "nonfinite_num": nonfinite_num,
                "total": total,
                "finite_ratio": finite_ratio,
            }
        return None

    @staticmethod
    def _find_nonfinite_log_vars(log_vars):
        bad_items = []
        if not isinstance(log_vars, dict):
            return bad_items
        for key, value in log_vars.items():
            try:
                value_f = float(value)
            except Exception:
                continue
            if not np.isfinite(value_f):
                bad_items.append((key, value_f))
        return bad_items

    def after_train_iter(self, runner) -> None:
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

        self.loss_scaler.scale(runner.outputs['loss']).backward()
        self.loss_scaler.unscale_(runner.optimizer)

        loss_nonfinite = not torch.isfinite(runner.outputs['loss']).all()
        grad_info = self._find_first_nonfinite_grad(runner.model)
        if grad_info is not None:
            runner.logger.warning(
                "[GradCheck] first non-finite grad: "
                f"name={grad_info['name']}, dtype={grad_info['dtype']}, shape={grad_info['shape']}, "
                f"nonfinite={grad_info['nonfinite_num']}/{grad_info['total']}, "
                f"finite_ratio={grad_info['finite_ratio']:.6f}, iter={runner.iter}"
            )
        if loss_nonfinite:
            runner.logger.warning(
                f"[GradCheck] non-finite total loss detected at iter={runner.iter}: "
                f"loss={float(runner.outputs['loss'])}"
            )

        bad_log_vars = self._find_nonfinite_log_vars(runner.outputs.get('log_vars', None))
        if len(bad_log_vars) > 0:
            bad_log_str = ", ".join([f"{k}={v}" for k, v in bad_log_vars])
            runner.logger.warning(
                f"[GradCheck] non-finite log_vars at iter={runner.iter}: {bad_log_str}"
            )

        # Important: if loss/grad is non-finite, skip this optimization step.
        # Avoid grad clipping with NaN/Inf grads, which can contaminate all grads.
        if loss_nonfinite or grad_info is not None:
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
            self.loss_scaler.update(self._scale_update_param)
            return

        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        # backward and update scaler
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()


def _eager_init_llava_before_optimizer(model, logger):
    """Initialize lazy LLaVA/LoRA modules before optimizer construction."""
    raw_model = model.module if hasattr(model, 'module') else model
    lazy_init_fn = getattr(raw_model, '_lazy_init_llava', None)
    if lazy_init_fn is None:
        return
    try:
        inited = bool(lazy_init_fn())
    except Exception as exc:
        logger.warning(f'Pre-optimizer llava eager init failed: {exc}')
        return
    if inited:
        logger.info('LLaVA eager init completed before optimizer build.')
    else:
        runtime_error = getattr(raw_model, '_llava_runtime_error', None)
        logger.warning(f'LLaVA eager init before optimizer returned False. runtime_error={runtime_error}')


def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
   
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    #assert len(dataset)==1s
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        ) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if eval_model is not None:
            eval_model = MMDataParallel(
                eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # build runner
    _eager_init_llava_before_optimizer(model, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    if eval_model is not None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHookWithGradCheck(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    # register profiler hook
    #trace_config = dict(type='tb_trace', dir_name='work_dir')
    #profiler_config = dict(on_trace_ready=trace_config)
    #runner.register_profiler_hook(profiler_config)
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
