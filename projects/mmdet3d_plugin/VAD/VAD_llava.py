import os
import json
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet.utils import get_root_logger
from PIL import Image

from projects.mmdet3d_plugin.VAD.VAD import VAD


class QFormerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, context_size=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        if context_size is None or int(context_size) == int(hidden_size):
            self.cross_attn = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.cross_attn = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=dropout,
                batch_first=True,
                kdim=int(context_size),
                vdim=int(context_size),
            )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, query_tokens, context_tokens, context_padding_mask=None):
        q = self.norm1(query_tokens)
        q_attn, _ = self.self_attn(q, q, q, need_weights=False)
        query_tokens = query_tokens + q_attn

        q = self.norm2(query_tokens)
        q_cross, _ = self.cross_attn(
            q,
            context_tokens,
            context_tokens,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + q_cross

        q = self.norm3(query_tokens)
        query_tokens = query_tokens + self.ffn(q)
        return query_tokens


class PlanningTokenAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        context_size=None,
        num_query_tokens=8,
        num_layers=2,
        num_heads=8,
        dropout=0.0,
        internal_dim=768,
    ):
        super().__init__()
        if int(internal_dim) % int(num_heads) != 0:
            raise ValueError(f"internal_dim ({internal_dim}) must be divisible by num_heads ({num_heads}).")

        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, internal_dim) * 0.02)
        if context_size is not None and int(context_size) != int(internal_dim):
            self.context_proj = nn.Linear(int(context_size), int(internal_dim))
        else:
            self.context_proj = nn.Identity()
            
        self.blocks = nn.ModuleList(
            [
                QFormerBlock(
                    internal_dim,
                    num_heads,
                    dropout=dropout,
                    context_size=internal_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(internal_dim)
        self.output_proj = nn.Linear(internal_dim, hidden_size)
    
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, context_tokens, context_padding_mask=None):
        batch_size = context_tokens.shape[0]
        
        # 把上次加的 context_norm 拿掉，恢復原狀
        context_tokens = self.context_proj(context_tokens)
        
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        for block in self.blocks:
            query_tokens = block(query_tokens, context_tokens, context_padding_mask)
            
        return self.output_proj(self.out_norm(query_tokens))


@DETECTORS.register_module()
class VADLLaVA(VAD):
    """VAD wrapper with planning-token adapters + LoRA on LLaVA LM."""

    def __init__(
        self,
        *args,
        llava_enabled=False,
        llava_model_name="llava-hf/llava-1.5-7b-hf",
        llava_device="cuda",
        llava_dtype="float16",
        llava_replace_ego_fut_preds=False,
        llava_freeze=True,
        llava_bev_topk=128,
        llava_max_agents=20,
        llava_max_map_vectors=20,
        llava_use_planning_adapter=True,
        llava_use_projector=True,
        llava_adapter_query_tokens=8,
        llava_adapter_num_layers=2,
        llava_adapter_num_heads=8,
        llava_adapter_dropout=0.0,
        llava_adapter_internal_dim=768,
        llava_use_lora=True,
        llava_lora_r=8,
        llava_lora_alpha=16,
        llava_lora_dropout=0.05,
        llava_lora_target_modules=None,
        llava_lora_modules_to_save=None,
        llava_hidden_size_hint=4096,
        llava_retry_init=False,
        llava_train_prompt="Predict future ego delta waypoints in ego frame.",
        llava_plan_loss_weight=1.0,
        llava_use_plan_constraint_loss=True,
        llava_checkpoint_mode="full",
        llava_gradient_checkpointing=True,
        drive_qa=False,
        llava_qa_loss_weight=1.0,
        llava_qa_prompt_template="USER: <image>\nQuestion: {question}\nASSISTANT:",
        llava_text_max_length=128,
        llava_image_pool_stride=2,
        llava_use_image=True,
        # === Stage C: TextDeltaPlanner (additive language-conditioned residual) ===
        text_delta_planner_enabled=False,
        text_delta_planner_only=False,        # if True, skip original LLaVA branch entirely
        text_delta_text_proj_dim=256,
        text_delta_hidden_dim=512,
        text_delta_max_length=64,
        text_delta_alpha_init=0.0,
        text_delta_rich_prompt=False,         # enrich instruction text with meta flags
        # FiLM variant of TextDeltaPlanner (per-channel γ,β modulation of ego_feats)
        text_delta_use_film=False,
        text_delta_film_use_gamma=True,
        text_delta_film_use_beta=True,
        text_delta_film_use_alpha=True,
        text_delta_film_use_concat=False,
        # Level-2 FDE recipe: auxiliary endpoint head with direct GT-t6s supervision
        text_delta_use_endpoint_head=False,
        text_delta_endpoint_pull_init=1.0,
        text_delta_endpoint_hidden_dim=256,
        text_delta_endpoint_loss_weight=1.0,
        # Goal-conditioned FiLM: predicted goal_xy modulates γ/β too
        text_delta_use_goal_conditioning=False,
        # Explicit Goal Prediction variant flags (user spec)
        text_delta_goal_head_text_only=False,
        text_delta_goal_concat_to_mlp=False,
        text_delta_endpoint_loss_type='l2',   # 'l2' or 'smooth_l1'
        # Agent-aware cross-attention
        text_delta_use_agent_attn=False,
        text_delta_agent_attn_dim=256,
        # Speed-class auxiliary head
        text_delta_use_speed_class_head=False,
        text_delta_speed_num_classes=7,
        text_delta_speed_class_loss_weight=0.05,
        # Speed bin edges (m/s) used to bucketize GT trajectory avg speed
        text_delta_speed_bin_edges=(1.0, 3.0, 6.0, 9.0, 12.0, 16.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.llava_enabled = llava_enabled
        # Stage C planner config (must come before _lazy_init_llava paths use them)
        self.text_delta_planner_enabled = bool(text_delta_planner_enabled)
        self.text_delta_planner_only = bool(text_delta_planner_only)
        self.text_delta_text_proj_dim = int(text_delta_text_proj_dim)
        self.text_delta_hidden_dim = int(text_delta_hidden_dim)
        self.text_delta_max_length = int(text_delta_max_length)
        self.text_delta_alpha_init = float(text_delta_alpha_init)
        self.text_delta_rich_prompt = bool(text_delta_rich_prompt)
        self.text_delta_use_film = bool(text_delta_use_film)
        self.text_delta_film_use_gamma = bool(text_delta_film_use_gamma)
        self.text_delta_film_use_beta = bool(text_delta_film_use_beta)
        self.text_delta_film_use_alpha = bool(text_delta_film_use_alpha)
        self.text_delta_film_use_concat = bool(text_delta_film_use_concat)
        self.text_delta_use_endpoint_head = bool(text_delta_use_endpoint_head)
        self.text_delta_endpoint_pull_init = float(text_delta_endpoint_pull_init)
        self.text_delta_endpoint_hidden_dim = int(text_delta_endpoint_hidden_dim)
        self.text_delta_endpoint_loss_weight = float(text_delta_endpoint_loss_weight)
        self.text_delta_use_goal_conditioning = bool(text_delta_use_goal_conditioning)
        self.text_delta_goal_head_text_only = bool(text_delta_goal_head_text_only)
        self.text_delta_goal_concat_to_mlp = bool(text_delta_goal_concat_to_mlp)
        self.text_delta_endpoint_loss_type = str(text_delta_endpoint_loss_type)
        self.text_delta_use_agent_attn = bool(text_delta_use_agent_attn)
        self.text_delta_agent_attn_dim = int(text_delta_agent_attn_dim)
        self.text_delta_use_speed_class_head = bool(text_delta_use_speed_class_head)
        self.text_delta_speed_num_classes = int(text_delta_speed_num_classes)
        self.text_delta_speed_class_loss_weight = float(text_delta_speed_class_loss_weight)
        self.text_delta_speed_bin_edges = tuple(float(e) for e in text_delta_speed_bin_edges)
        self._text_delta_planner = None       # nn.Module lazy-built after LLaVA loaded
        self.llava_model_name = llava_model_name
        self.llava_device = llava_device
        self.llava_dtype = llava_dtype
        self.llava_replace_ego_fut_preds = llava_replace_ego_fut_preds
        self.llava_freeze = llava_freeze
        self.llava_bev_topk = llava_bev_topk
        self.llava_max_agents = llava_max_agents
        self.llava_max_map_vectors = llava_max_map_vectors

        self.llava_use_planning_adapter = llava_use_planning_adapter
        self.llava_use_projector = llava_use_projector
        self.llava_adapter_query_tokens = llava_adapter_query_tokens
        self.llava_adapter_num_layers = llava_adapter_num_layers
        self.llava_adapter_num_heads = llava_adapter_num_heads
        self.llava_adapter_dropout = llava_adapter_dropout
        self.llava_adapter_internal_dim = llava_adapter_internal_dim

        self.llava_use_lora = llava_use_lora
        self.llava_lora_r = llava_lora_r
        self.llava_lora_alpha = llava_lora_alpha
        self.llava_lora_dropout = llava_lora_dropout
        self.llava_lora_target_modules = llava_lora_target_modules
        self.llava_lora_modules_to_save = llava_lora_modules_to_save
        self.llava_hidden_size_hint = llava_hidden_size_hint
        self.llava_retry_init = llava_retry_init
        self.llava_train_prompt = llava_train_prompt
        self.llava_plan_loss_weight = llava_plan_loss_weight
        self.llava_use_plan_constraint_loss = llava_use_plan_constraint_loss
        self.llava_checkpoint_mode = llava_checkpoint_mode
        self.llava_gradient_checkpointing = llava_gradient_checkpointing
        self.drive_qa = drive_qa
        self.llava_qa_loss_weight = llava_qa_loss_weight
        self.llava_qa_prompt_template = llava_qa_prompt_template
        self.llava_text_max_length = llava_text_max_length
        self.llava_image_pool_stride = llava_image_pool_stride
        self.llava_use_image = llava_use_image

        self._llava_model = None
        self._llava_processor = None
        self._llava_tokenizer = None
        self._llava_hidden_size = llava_hidden_size_hint
        self._llava_runtime_error = None
        self._llava_peft_error = None
        self._llava_lora_applied = False
        self._llava_train_modules_initialized = False
        self._llava_init_state = "not_started"
        self._logger = get_root_logger()
        self._log_once_keys = set()

        self._token_projectors = nn.ModuleDict()
        self._planning_adapters = nn.ModuleDict()
        self._planning_type_bias = nn.ParameterDict()
        self._llava_plan_head = None

        if (not self.llava_use_projector) and (not self.llava_use_planning_adapter) \
                and not self.text_delta_planner_enabled:
            raise ValueError(
                "When llava_use_projector=False, llava_use_planning_adapter must be True "
                "(or text_delta_planner_enabled=True for Stage C).")

        # Stage C: eager-init LLaVA + TextDeltaPlanner so optimizer sees all
        # trainable params at build time. Without this the lazy init only fires
        # on first forward (which is AFTER optimizer construction) → new params
        # have no LR/momentum/state in the optimizer.
        if self.text_delta_planner_enabled:
            self._lazy_init_llava()
        if self.llava_checkpoint_mode not in {"full", "no_llava_base", "adapter_only"}:
            raise ValueError(
                "llava_checkpoint_mode must be one of {'full', 'no_llava_base', 'adapter_only'}."
            )

        # Important: do not load LLaVA in __init__.
        # Keep DDP init memory low; load LLaVA lazily in first forward.
        self._lazy_init_planning_adapters(self._llava_hidden_size)
        self._lazy_init_train_modules()
        # Enforce checkpoint filtering for MMCV paths that may not call
        # the overridden `state_dict()` directly.
        self._register_state_dict_hook(self._checkpoint_state_dict_hook)

    @staticmethod
    def _strip_state_prefix(key, prefix):
        if prefix and key.startswith(prefix):
            return key[len(prefix):]
        return key

    def _is_llava_adapter_state_key(self, local_key):
        return (
            local_key.startswith("_planning_adapters.")
            or local_key.startswith("_planning_type_bias.")
            or local_key.startswith("_llava_plan_head.")
            or local_key.startswith("_token_projectors.")
        )

    def _is_llava_lora_state_key(self, local_key):
        if not local_key.startswith("_llava_model."):
            return False
        if "lora_" in local_key:
            return True
        if self.llava_lora_modules_to_save is not None:
            for module_name in self.llava_lora_modules_to_save:
                if f".{module_name}." in local_key:
                    return True
        return False

    def _keep_ckpt_key(self, local_key):
        if self.llava_checkpoint_mode == "full":
            return True
        if self.llava_checkpoint_mode == "no_llava_base":
            if local_key.startswith("_llava_model.") and (not self._is_llava_lora_state_key(local_key)):
                return False
            return True
        # adapter_only
        return self._is_llava_adapter_state_key(local_key) or self._is_llava_lora_state_key(local_key)

    def _checkpoint_state_dict_hook(self, module, state_dict, prefix, local_metadata):
        if self.llava_checkpoint_mode == "full":
            return state_dict
        for key in list(state_dict.keys()):
            local_key = self._strip_state_prefix(key, prefix)
            if not self._keep_ckpt_key(local_key):
                state_dict.pop(key)
        return state_dict

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.llava_checkpoint_mode == "full":
            return state

        filtered = OrderedDict()
        for key, value in state.items():
            local_key = self._strip_state_prefix(key, prefix)
            if self._keep_ckpt_key(local_key):
                filtered[key] = value

        return filtered

    @staticmethod
    def _cast_module_fp32(module):
        if module is None:
            return
        module.to(dtype=torch.float32)

    @staticmethod
    def _cast_parameter_dict_fp32(param_dict):
        if param_dict is None:
            return
        for _, param in param_dict.items():
            if param.dtype != torch.float32:
                param.data = param.data.to(dtype=torch.float32)

    def _is_rank0(self):
        return (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0)

    def _log_info_once(self, key, msg):
        if key in self._log_once_keys:
            return
        if self._is_rank0():
            self._logger.info(msg)
        self._log_once_keys.add(key)

    def _log_warning_once(self, key, msg):
        if key in self._log_once_keys:
            return
        if self._is_rank0():
            self._logger.warning(msg)
        self._log_once_keys.add(key)

    def _sanitize_tensor(self, tensor, name, clamp_val=None):
        if tensor is None or (not torch.is_tensor(tensor)):
            return tensor
        finite_mask = torch.isfinite(tensor)
        if not finite_mask.all():
            finite_ratio = finite_mask.float().mean().item()
            self._log_warning_once(
                f"{name}_non_finite",
                (
                    f"[VADLLaVA] non-finite tensor at {name}: "
                    f"finite_ratio={finite_ratio:.6f}, dtype={tensor.dtype}, shape={tuple(tensor.shape)}. "
                    "Apply nan_to_num."
                ),
            )
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_val is not None:
            tensor = torch.clamp(tensor, min=-float(clamp_val), max=float(clamp_val))
        return tensor

    def train(self, mode=True):
        super().train(mode)
        # Runner will call model.train() every epoch; keep frozen LLaVA in eval mode.
        if self.llava_freeze and self._llava_model is not None:
            self._llava_model.eval()
        # Stage C planner-only: img_backbone / pts_bbox_head are lr_mult=0 frozen
        # but train() mode still updates BN running stats every forward → over
        # 60 ep this drifts and breaks bit-identity vs the load_from baseline at
        # eval time (observed +0.105 m no-lang ADE@6s artifact). Pin them to
        # eval() so BN running mean/var stay locked.
        if mode and getattr(self, 'text_delta_planner_only', False):
            for mod_name in ('img_backbone', 'img_neck', 'pts_bbox_head'):
                mod = getattr(self, mod_name, None)
                if mod is not None:
                    mod.eval()
        return self

    def _to_dtype(self, name: str):
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(name.lower(), torch.float16)

    def _get_model_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        model_device = None
        if self._llava_model is not None and hasattr(self._llava_model, "device"):
            model_device = self._llava_model.device
        if model_device is None or str(model_device) == "meta":
            model_device = next(self._llava_model.parameters()).device
        model_dtype = next(self._llava_model.parameters()).dtype
        return model_device, model_dtype

    def _apply_lora_to_output(self):
        if not self.llava_use_lora:
            return True
        if self._llava_lora_applied:
            return True

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            self._llava_peft_error = f"Failed to import peft: {exc}"
            return False

        target_modules = self.llava_lora_target_modules
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.llava_lora_r,
            lora_alpha=self.llava_lora_alpha,
            lora_dropout=self.llava_lora_dropout,
            target_modules=list(target_modules),
            modules_to_save=self.llava_lora_modules_to_save,
            bias="none",
        )
        self._llava_model.language_model = get_peft_model(
            self._llava_model.language_model,
            lora_config,
        )
        self._llava_lora_applied = True
        return True

    def _freeze_llava_base(self):
        if self._llava_model is None or not self.llava_freeze:
            return

        self._llava_model.eval()
        for _, param in self._llava_model.named_parameters():
            param.requires_grad = False

        if self.llava_use_lora and self._llava_lora_applied:
            for name, param in self._llava_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            if self.llava_lora_modules_to_save is not None:
                for name, param in self._llava_model.named_parameters():
                    for module_name in self.llava_lora_modules_to_save:
                        if module_name in name:
                            param.requires_grad = True

    def _cast_lora_trainable_to_fp32(self):
        if self._llava_model is None or not self.llava_use_lora or not self._llava_lora_applied:
            return
        for name, param in self._llava_model.named_parameters():
            is_lora = "lora_" in name
            is_modules_to_save = False
            if self.llava_lora_modules_to_save is not None:
                for module_name in self.llava_lora_modules_to_save:
                    if module_name in name:
                        is_modules_to_save = True
                        break
            if (is_lora or is_modules_to_save) and param.dtype != torch.float32:
                param.data = param.data.to(dtype=torch.float32)

    def _enable_llava_gradient_checkpointing(self):
        if self._llava_model is None or not self.llava_gradient_checkpointing:
            return
        modules = [self._llava_model, getattr(self._llava_model, "language_model", None)]
        for module in modules:
            if module is None:
                continue
            grad_ckpt_fn = getattr(module, "gradient_checkpointing_enable", None)
            if callable(grad_ckpt_fn):
                try:
                    grad_ckpt_fn()
                except TypeError:
                    grad_ckpt_fn({})
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False
            enable_grads = getattr(module, "enable_input_require_grads", None)
            if callable(enable_grads):
                try:
                    enable_grads()
                except Exception:
                    pass

    def _planning_context_dims(self):
        head = self.pts_bbox_head
        map_pts = int(getattr(head, "map_num_pts_per_vec", 20))
        code_size = int(getattr(head, "code_size", 10))
        ego_dim = None
        ego_decoder = getattr(head, "ego_fut_decoder", None)
        if ego_decoder is not None:
            modules = ego_decoder if isinstance(ego_decoder, nn.Sequential) else [ego_decoder]
            for layer in modules:
                if isinstance(layer, nn.Linear):
                    ego_dim = int(layer.in_features)
                    break
        if ego_dim is None:
            ego_lcf_feat_idx = getattr(head, "ego_lcf_feat_idx", None)
            ego_dim = int(getattr(head, "embed_dims", 256)) * 2
            if ego_lcf_feat_idx is not None:
                ego_dim += len(ego_lcf_feat_idx)
        return {
            "bev": int(getattr(head, "embed_dims", 256)),
            "agent": code_size + 1,
            "map": map_pts * 2 + 1,
            "ego": ego_dim,
        }

    @staticmethod
    def _match_last_dim(tokens, target_dim):
        if tokens is None:
            return None
        cur_dim = int(tokens.shape[-1])
        target_dim = int(target_dim)
        if cur_dim == target_dim:
            return tokens
        if cur_dim > target_dim:
            return tokens[..., :target_dim]
        pad_shape = list(tokens.shape[:-1]) + [target_dim - cur_dim]
        pad = torch.zeros(pad_shape, device=tokens.device, dtype=tokens.dtype)
        return torch.cat([tokens, pad], dim=-1)

    def _lazy_init_planning_adapters(self, hidden_size):
        if not self.llava_use_planning_adapter:
            return
        if len(self._planning_adapters) > 0:
            return

        planning_dims = self._planning_context_dims()
        for token_type in ["bev", "agent", "map", "ego"]:
            context_size = hidden_size if self.llava_use_projector else planning_dims[token_type]
            adapter = PlanningTokenAdapter(
                hidden_size=hidden_size,
                context_size=context_size,
                num_query_tokens=self.llava_adapter_query_tokens,
                num_layers=self.llava_adapter_num_layers,
                num_heads=self.llava_adapter_num_heads,
                dropout=self.llava_adapter_dropout,
                internal_dim=self.llava_adapter_internal_dim,
            )
            self._planning_adapters[token_type] = adapter.to(dtype=torch.float32)
            bias = nn.Parameter(torch.zeros(1, 1, hidden_size, dtype=torch.float32))
            nn.init.normal_(bias, std=0.02)
            self._planning_type_bias[token_type] = bias

    def _lazy_init_llava(self):
        if not (self.llava_enabled or self.drive_qa):
            return False
        if self._llava_init_state == "ok":
            return True
        if self._llava_init_state == "failed" and not self.llava_retry_init:
            self._log_warning_once(
                "llava_init_skip_retry",
                f"[VADLLaVA] Skip retry after init failed. last_error={self._llava_runtime_error}",
            )
            return False
        if self._llava_model is not None and self._llava_processor is not None:
            self._llava_init_state = "ok"
            return True
        self._llava_init_state = "in_progress"
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as exc:
            self._llava_runtime_error = f"Failed to import transformers LLaVA: {exc}"
            self._llava_init_state = "failed"
            self._log_warning_once("llava_import_failed", f"[VADLLaVA] {self._llava_runtime_error}")
            return False

        try:
            dtype = self._to_dtype(self.llava_dtype)
            load_kwargs = {"torch_dtype": dtype}
            if self.llava_device in {"auto", "balanced", "balanced_low_0", "sequential"}:
                load_kwargs["device_map"] = self.llava_device
            self._llava_model = LlavaForConditionalGeneration.from_pretrained(
                self.llava_model_name,
                **load_kwargs,
            )
            if "device_map" not in load_kwargs:
                self._llava_model.to(self.llava_device)
            self._llava_processor = AutoProcessor.from_pretrained(self.llava_model_name)
            self._llava_tokenizer = self._llava_processor.tokenizer

            loaded_hidden = self._llava_model.language_model.config.hidden_size
            if self._llava_hidden_size is None:
                self._llava_hidden_size = loaded_hidden
            elif int(self._llava_hidden_size) != int(loaded_hidden):
                self._llava_runtime_error = (
                    f"Hidden size mismatch: hint={self._llava_hidden_size}, loaded={loaded_hidden}. "
                    "Please set llava_hidden_size_hint to the model hidden size."
                )
                self._llava_init_state = "failed"
                return False
            self._lazy_init_planning_adapters(self._llava_hidden_size)
            self._lazy_init_train_modules()
            lora_ok = self._apply_lora_to_output()
            if not lora_ok:
                self._llava_runtime_error = self._llava_peft_error
                self._llava_init_state = "failed"
                self._log_warning_once("llava_lora_failed", f"[VADLLaVA] {self._llava_runtime_error}")
                return False
            self._enable_llava_gradient_checkpointing()
            self._freeze_llava_base()
            self._cast_lora_trainable_to_fp32()
            self._llava_runtime_error = None
            self._llava_init_state = "ok"
            self._log_info_once(
                "llava_init_ok",
                f"[VADLLaVA] LLaVA init done on {self.llava_device} (hidden={self._llava_hidden_size}).",
            )
            return True
        except Exception as exc:
            self._llava_runtime_error = f"Failed to load LLaVA model '{self.llava_model_name}': {exc}"
            self._llava_model = None
            self._llava_processor = None
            self._llava_tokenizer = None
            self._llava_init_state = "failed"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._log_warning_once("llava_load_failed", f"[VADLLaVA] {self._llava_runtime_error}")
            return False

    def _lazy_init_train_modules(self):
        if self._llava_train_modules_initialized:
            return
        if self._llava_hidden_size is None:
            return

        head = self.pts_bbox_head
        hidden_size = self._llava_hidden_size
        ego_mode = int(getattr(head, "ego_fut_mode", 3))
        fut_ts = int(getattr(head, "fut_ts", self.fut_ts))
        # v7: when soft cmd routing is on, LLaVA plan head also emits a
        # single trajectory (no per-cmd mode). Effective mode dim = 1.
        soft_routing = bool(getattr(head, "enable_soft_cmd_routing", False))
        out_mode = 1 if soft_routing else ego_mode
        self._llava_plan_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_mode * fut_ts * 2),
        )
        self._cast_module_fp32(self._planning_adapters)
        self._cast_parameter_dict_fp32(self._planning_type_bias)
        self._cast_module_fp32(self._llava_plan_head)

        # Stage C: build TextDeltaPlanner once LLaVA tokenizer is ready.
        if self.text_delta_planner_enabled and self._text_delta_planner is None \
                and self._llava_tokenizer is not None:
            ego_feat_dim = int(getattr(self, 'text_delta_ego_feat_dim', 512) or 512)
            if self.text_delta_use_film:
                from projects.mmdet3d_plugin.VAD.text_delta_planner_film import (
                    TextDeltaPlannerFiLM,
                )
                self._text_delta_planner = TextDeltaPlannerFiLM(
                    llava_tokenizer=self._llava_tokenizer,
                    ego_feat_dim=ego_feat_dim,
                    text_hidden_size=int(hidden_size),
                    ego_fut_mode=int(ego_mode),
                    fut_ts=int(fut_ts),
                    text_proj_dim=self.text_delta_text_proj_dim,
                    mlp_hidden_dim=self.text_delta_hidden_dim,
                    text_max_length=self.text_delta_max_length,
                    alpha_init=self.text_delta_alpha_init,
                    use_gamma=self.text_delta_film_use_gamma,
                    use_beta=self.text_delta_film_use_beta,
                    use_alpha=self.text_delta_film_use_alpha,
                    use_concat=self.text_delta_film_use_concat,
                    use_endpoint_head=self.text_delta_use_endpoint_head,
                    endpoint_pull_init=self.text_delta_endpoint_pull_init,
                    endpoint_hidden_dim=self.text_delta_endpoint_hidden_dim,
                    use_goal_conditioning=self.text_delta_use_goal_conditioning,
                    goal_head_text_only=self.text_delta_goal_head_text_only,
                    goal_concat_to_mlp=self.text_delta_goal_concat_to_mlp,
                    use_agent_attn=self.text_delta_use_agent_attn,
                    agent_attn_dim=self.text_delta_agent_attn_dim,
                    use_speed_class_head=self.text_delta_use_speed_class_head,
                    speed_num_classes=self.text_delta_speed_num_classes,
                )
            else:
                from projects.mmdet3d_plugin.VAD.text_delta_planner import TextDeltaPlanner
                self._text_delta_planner = TextDeltaPlanner(
                    llava_tokenizer=self._llava_tokenizer,
                    ego_feat_dim=ego_feat_dim,
                    text_hidden_size=int(hidden_size),
                    ego_fut_mode=int(ego_mode),
                    fut_ts=int(fut_ts),
                    text_proj_dim=self.text_delta_text_proj_dim,
                    mlp_hidden_dim=self.text_delta_hidden_dim,
                    text_max_length=self.text_delta_max_length,
                    alpha_init=self.text_delta_alpha_init,
                )
            self._cast_module_fp32(self._text_delta_planner)
            # Explicit move to llava device so MMDataParallel's pre-check
            # (all params on device_ids[0]) passes during eval setup.
            self._text_delta_planner.to(self.llava_device)
            self._log_warning_once(
                "text_delta_planner_built",
                f"[VADLLaVA] TextDeltaPlanner built (text_proj={self.text_delta_text_proj_dim}, "
                f"mlp_hidden={self.text_delta_hidden_dim}, ego_fut_mode={ego_mode}, fut_ts={fut_ts}).",
            )

        self._llava_train_modules_initialized = True

    def _get_llava_language_model(self):
        """Return the underlying causal-LM module (for text-only forwards)."""
        if self._llava_model is None:
            return None
        if hasattr(self._llava_model, 'language_model'):
            return self._llava_model.language_model
        # PEFT-wrapped model may expose .base_model.model.language_model
        bm = getattr(self._llava_model, 'base_model', None)
        if bm is not None and hasattr(bm, 'model') and hasattr(bm.model, 'language_model'):
            return bm.model.language_model
        return None

    def _build_rich_text(self, meta) -> str:
        """Compose richer instruction text from doScenes meta flags.

        Includes the raw instruction plus three small annotations from
        LoadDoScenesInstruction (no future leak — all from the same anchor
        frame's annotation):
          - intent type (S/D)
          - has_static_reference (bool)
          - has_dynamic_reference (bool)
        """
        if not isinstance(meta, dict):
            return ''
        inst = meta.get('ego_instruction', '') or ''
        if not isinstance(inst, str) or not inst.strip():
            return ''
        inst = inst.strip()
        itype = meta.get('ego_instruction_type', '') or ''
        if isinstance(itype, str):
            itype = itype.strip().upper()
        has_static = bool(meta.get('has_static_reference', False))
        has_dynamic = bool(meta.get('has_dynamic_reference', False))
        parts = [f"Instruction: {inst}"]
        if itype in ('S', 'D'):
            parts.append(f"Type: {'static-intent' if itype == 'S' else 'dynamic-intent'}")
        parts.append(f"Static reference: {'yes' if has_static else 'no'}")
        parts.append(f"Dynamic reference: {'yes' if has_dynamic else 'no'}")
        return ". ".join(parts) + "."

    def _build_instruction_text(self, meta) -> str:
        """Single sample → text string, gated by self.text_delta_rich_prompt."""
        if self.text_delta_rich_prompt:
            return self._build_rich_text(meta)
        if isinstance(meta, dict):
            return (meta.get('ego_instruction', '') or '') if meta else ''
        return ''

    def _apply_text_delta(self, container, ego_instructions):
        """Add alpha * delta to container['ego_fut_preds'] if planner is ready.
        Mutates `container` in-place. No-op when planner disabled or instructions empty."""
        if not self.text_delta_planner_enabled:
            return
        # Ensure LLaVA is loaded and train modules (incl. TextDeltaPlanner) built.
        # Without this, text_delta_planner_only=True short-circuits the usual
        # _lazy_init_llava callers and the planner never gets registered.
        if self._text_delta_planner is None:
            if not self._lazy_init_llava():
                return
        planner = self._text_delta_planner
        if planner is None:
            return
        if not isinstance(container, dict):
            return
        if 'ego_fut_preds' not in container or 'ego_feats' not in container:
            return
        if not isinstance(ego_instructions, (list, tuple)) or len(ego_instructions) == 0:
            return
        lm = self._get_llava_language_model()
        if lm is None:
            return
        # Optional agent_query/agent_mask passthrough for agent-aware attn.
        # In eval path bbox_result was stripped of the batch dim (outs[k][i]) →
        # add it back so planner sees [B, num_agents, D] / [B, num_agents].
        ag_q = container.get('agent_query', None)
        ag_m = container.get('agent_mask', None)
        if ag_q is not None and ag_q.dim() == 2:
            ag_q = ag_q.unsqueeze(0)
        if ag_m is not None and ag_m.dim() == 1:
            ag_m = ag_m.unsqueeze(0)
        delta = planner(
            ego_instructions, container['ego_feats'], lm,
            agent_query=ag_q,
            agent_mask=ag_m,
        )
        # Match efp dim/shape: efp can be (B, M, T, 2) in train or (M, T, 2)
        # in test (after VADLLaVA strips batch dim via _attach_llava_result).
        # Planner outputs (B, M, T, 2). Squeeze to match efp's dim count.
        efp = container['ego_fut_preds']
        if efp.dim() == 3 and delta.dim() == 4 and delta.shape[0] == 1:
            delta = delta.squeeze(0)
        # Also align device (efp in test may be on cpu after .cpu() conversion).
        container['ego_fut_preds'] = efp + delta.to(
            device=efp.device, dtype=efp.dtype)
        # Expose endpoint head's prediction so the train loss can supervise it
        # directly with GT t=6s position.
        if getattr(planner, 'last_endpoint', None) is not None:
            container['text_delta_endpoint'] = planner.last_endpoint
            container['text_delta_endpoint_pull'] = planner.endpoint_pull
        # Expose speed-class logits + keep_idx for auxiliary CE loss
        if getattr(planner, 'last_speed_logits', None) is not None:
            container['text_delta_speed_logits'] = planner.last_speed_logits
            container['text_delta_speed_keep_idx'] = planner.last_speed_keep_idx

    @staticmethod
    def _unwrap_img_meta(meta):
        while isinstance(meta, dict) and len(meta) == 1 and 0 in meta:
            meta = meta[0]
        return meta

    def _extract_image_paths_from_meta(self, meta) -> List[str]:
        meta = self._unwrap_img_meta(meta)
        for key in ("filename", "img_filename"):
            value = meta.get(key, None) if isinstance(meta, dict) else None
            if isinstance(value, str) and value:
                return [value]
            if isinstance(value, (list, tuple)):
                return [path for path in value if isinstance(path, str) and path]
        return []

    def _open_llava_image(self, image_path: str):
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as exc:
            self._log_warning_once(
                f"llava_image_open_failed_{image_path}",
                f"[VADLLaVA] Failed to open image '{image_path}': {exc}",
            )
            return None

    def _build_llava_image_mosaic(self, image_paths: List[str]):
        images = [self._open_llava_image(path) for path in image_paths]
        images = [img for img in images if img is not None]
        if not images:
            return Image.new("RGB", (336, 336), color=(0, 0, 0))
        if len(images) == 1:
            return images[0].resize((336, 336), Image.BICUBIC)

        cell_size = (336, 336)
        cols = 3 if len(images) > 2 else 2
        rows = (len(images) + cols - 1) // cols
        canvas = Image.new("RGB", (cell_size[0] * cols, cell_size[1] * rows), color=(0, 0, 0))
        for idx, img in enumerate(images):
            img = img.resize(cell_size, Image.BICUBIC)
            x = (idx % cols) * cell_size[0]
            y = (idx // cols) * cell_size[1]
            canvas.paste(img, (x, y))
        return canvas

    def _denormalize_image_tensor(self, img_tensor: torch.Tensor, img_meta: Optional[Dict[str, Any]] = None):
        if img_tensor is None or not torch.is_tensor(img_tensor):
            return None

        tensor = img_tensor.detach().float().cpu()
        if tensor.dim() != 3 or tensor.shape[0] not in (1, 3):
            return None

        meta = self._unwrap_img_meta(img_meta) if isinstance(img_meta, dict) else img_meta
        norm_cfg = meta.get("img_norm_cfg", {}) if isinstance(meta, dict) else {}
        mean = torch.tensor(norm_cfg.get("mean", [0.0, 0.0, 0.0]), dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(norm_cfg.get("std", [1.0, 1.0, 1.0]), dtype=torch.float32).view(-1, 1, 1)

        if tensor.shape[0] == 1:
            mean = mean[:1]
            std = std[:1]
        tensor = tensor * std + mean
        tensor = tensor.clamp(0.0, 255.0)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        array = tensor.permute(1, 2, 0).numpy().round().astype("uint8")
        return Image.fromarray(array)

    def _build_llava_image_mosaic_from_tensor(self, img_tensor, img_meta=None):
        if img_tensor is None or not torch.is_tensor(img_tensor):
            return None

        tensor = img_tensor
        if tensor.dim() == 5:
            tensor = tensor[0]
        if tensor.dim() != 4:
            return None

        if isinstance(img_meta, list) and len(img_meta) == tensor.shape[0]:
            meta_list = img_meta
        else:
            meta_list = [img_meta for _ in range(tensor.shape[0])]

        images = []
        for cam_idx in range(tensor.shape[0]):
            pil_img = self._denormalize_image_tensor(tensor[cam_idx], meta_list[cam_idx])
            if pil_img is not None:
                images.append(pil_img)
        if not images:
            return None
        if len(images) == 1:
            return images[0].resize((336, 336), Image.BICUBIC)

        cell_size = (336, 336)
        cols = 3 if len(images) > 2 else 2
        rows = (len(images) + cols - 1) // cols
        canvas = Image.new("RGB", (cell_size[0] * cols, cell_size[1] * rows), color=(0, 0, 0))
        for idx, img in enumerate(images):
            img = img.resize(cell_size, Image.BICUBIC)
            x = (idx % cols) * cell_size[0]
            y = (idx // cols) * cell_size[1]
            canvas.paste(img, (x, y))
        return canvas

    def _build_llava_images_from_metas(self, img_metas, img=None):
        if img_metas is None:
            return None
        images = []
        img_tensor = img
        if torch.is_tensor(img_tensor) and img_tensor.dim() == 5:
            per_sample_imgs = [img_tensor[i] for i in range(img_tensor.shape[0])]
        else:
            per_sample_imgs = [None for _ in range(len(img_metas))]

        for sample_idx, meta in enumerate(img_metas):
            sample_img = per_sample_imgs[sample_idx] if sample_idx < len(per_sample_imgs) else None
            mosaic = self._build_llava_image_mosaic_from_tensor(sample_img, meta)
            if mosaic is None:
                image_paths = self._extract_image_paths_from_meta(meta)
                mosaic = self._build_llava_image_mosaic(image_paths)
            images.append(mosaic)
        return images

    @staticmethod
    def _format_llava_prompt(prompt: str, include_image: bool) -> str:
        prompt = (prompt or "").strip()
        if include_image:
            if "<image>" not in prompt:
                prompt = f"<image>\n{prompt}"
            if "USER:" not in prompt:
                prompt = f"USER: {prompt}"
        elif "USER:" not in prompt:
            prompt = f"USER: {prompt}"

        if "ASSISTANT:" not in prompt:
            prompt = f"{prompt}\nASSISTANT:"
        return prompt

    def _processor_call(self, texts, images=None, padding=True, truncation=True):
        processor_kwargs = dict(
            text=texts,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )
        if self.llava_text_max_length is not None:
            processor_kwargs["max_length"] = int(self.llava_text_max_length)
        if images is None:
            return self._llava_processor(**processor_kwargs)
        processor_kwargs["images"] = images
        return self._llava_processor(**processor_kwargs)

    @staticmethod
    def _move_processor_batch_to_device(batch, device, dtype):
        moved = {}
        for key, value in batch.items():
            if not torch.is_tensor(value):
                moved[key] = value
                continue
            if key == "pixel_values":
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        return moved

    def _mosaic_tensor_to_pil(self, mosaic_tensor):
        if mosaic_tensor is None or not torch.is_tensor(mosaic_tensor):
            return None
        tensor = mosaic_tensor.detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() != 3:
            return None
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array)

    def _build_llava_images_from_mosaic_batch(self, llava_mosaic_img):
        if isinstance(llava_mosaic_img, (list, tuple)) and len(llava_mosaic_img) > 0:
            llava_mosaic_img = llava_mosaic_img[0]
        if llava_mosaic_img is None or not torch.is_tensor(llava_mosaic_img):
            return None
        if llava_mosaic_img.dim() == 3:
            llava_mosaic_img = llava_mosaic_img.unsqueeze(0)
        return [self._mosaic_tensor_to_pil(llava_mosaic_img[i]) for i in range(llava_mosaic_img.shape[0])]

    def _resolve_llava_images(self, llava_mosaic_img):
        if not self.llava_use_image:
            return None
        images = self._build_llava_images_from_mosaic_batch(llava_mosaic_img)
        if images is None:
            self._log_warning_once(
                "llava_image_missing",
                "[VADLLaVA] llava_use_image=True but llava_mosaic_img is missing. Fall back to text-only input.",
            )
        return images

    def _pool_llava_image_features(self, image_features):
        if image_features is None or image_features.dim() != 3:
            return image_features

        stride = max(1, int(self.llava_image_pool_stride))
        if stride <= 1 or image_features.shape[1] <= 1:
            return image_features

        batch_size, num_tokens, hidden_size = image_features.shape
        side = int(round(num_tokens ** 0.5))
        pooled = None
        if side * side == num_tokens and side >= stride:
            spatial = image_features.reshape(batch_size, side, side, hidden_size)
            spatial = spatial.permute(0, 3, 1, 2).contiguous().float()
            pooled = F.avg_pool2d(spatial, kernel_size=stride, stride=stride, ceil_mode=False)
            pooled = pooled.permute(0, 2, 3, 1).reshape(batch_size, -1, hidden_size)
        elif num_tokens >= stride:
            temporal = image_features.transpose(1, 2).contiguous().float()
            pooled = F.avg_pool1d(temporal, kernel_size=stride, stride=stride, ceil_mode=False)
            pooled = pooled.transpose(1, 2).contiguous()

        if pooled is None:
            return image_features

        self._log_info_once(
            f"llava_image_pool_{stride}",
            f"[VADLLaVA] Pool LLaVA image tokens with stride={stride}: {num_tokens} -> {pooled.shape[1]}",
        )
        return pooled.to(dtype=image_features.dtype)

    def _prepare_llava_image_features(self, device, dtype, llava_mosaic_img=None, images=None):
        if images is None:
            images = self._resolve_llava_images(llava_mosaic_img)
        if images is None:
            return None
        image_processor = getattr(self._llava_processor, "image_processor", None)
        if image_processor is not None:
            image_inputs = image_processor(images=images, return_tensors="pt")
        else:
            image_inputs = self._llava_processor(images=images, return_tensors="pt")
        image_inputs = self._move_processor_batch_to_device(image_inputs, device, dtype)
        pixel_values = image_inputs.get("pixel_values", None)
        if pixel_values is None:
            return None
        image_sizes = image_inputs.get("image_sizes", None)
        need_grad = self.training and self._llava_vision_requires_grad()
        with torch.set_grad_enabled(need_grad):
            image_features = self._extract_llava_image_features(
                model=self._llava_model,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
        if not need_grad:
            image_features = image_features.detach()
        image_features = self._pool_llava_image_features(image_features)
        return image_features.to(device=device, dtype=dtype)

    def _tokenize_llava_prompts(self, prompts, device, dtype):
        input_pack = self._llava_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.llava_text_max_length) if self.llava_text_max_length is not None else None,
        )
        input_ids = input_pack["input_ids"].to(device)
        attention_mask = input_pack["attention_mask"].to(device)
        inputs_embeds = self._llava_model.get_input_embeddings()(input_ids).to(dtype=dtype)
        return input_ids, attention_mask, inputs_embeds
    
    def _compute_causal_lm_loss_from_hidden(self, lm_head, hidden, labels):
        if lm_head is None or hidden is None or labels is None:
            return None
        shift_hidden = hidden[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return shift_hidden.sum() * 0.0

        valid_hidden = shift_hidden[valid_mask]
        valid_labels = shift_labels[valid_mask]
        logits = lm_head(valid_hidden)

        # pred_ids = logits.argmax(dim=-1)
        # tokenizer = self._llava_tokenizer

        # print("pred text:", tokenizer.decode(pred_ids[:50], skip_special_tokens=False))
        # print("gt text:", tokenizer.decode(valid_labels[:50], skip_special_tokens=False))
        return F.cross_entropy(logits.float(), valid_labels.view(-1))

    @staticmethod
    def _compute_causal_lm_loss(logits, labels):
        if logits is None or labels is None:
            return None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _get_lm_hidden_backbone(lm_model):
        module = lm_model
        for _ in range(4):
            next_module = None
            model_attr = getattr(module, "model", None)
            if model_attr is not None and model_attr is not module:
                next_module = model_attr
            base_model_attr = getattr(module, "base_model", None)
            if next_module is None and base_model_attr is not None and base_model_attr is not module:
                next_module = base_model_attr
            if next_module is None:
                break
            module = next_module
        return module

    def _llava_vision_requires_grad(self):
        if self._llava_model is None:
            return False
        modules = [
            getattr(self._llava_model, "vision_tower", None),
            getattr(self._llava_model, "multi_modal_projector", None),
        ]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                if param.requires_grad:
                    return True
        return False

    def _forward_lm_last_hidden(self, lm_model, inputs_embeds, attention_mask):
        forward_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        backbone = self._get_lm_hidden_backbone(lm_model)
        try:
            lm_out = backbone(**forward_kwargs)
            hidden = getattr(lm_out, "last_hidden_state", None)
            if hidden is not None:
                return hidden
        except (AttributeError, TypeError):
            pass

        lm_out = lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = getattr(lm_out, "hidden_states", None)
        if hidden_states is None or len(hidden_states) == 0:
            raise RuntimeError("LLaVA language model did not return hidden states.")
        return hidden_states[-1]

    def _expand_labels_with_image_tokens(
        self,
        input_ids,
        labels,
        image_features,
        merged_seq_len,
        attention_mask=None,
    ):
        if labels is None:
            return None
        image_token_index = getattr(self._llava_model.config, "image_token_index", None)
        if image_token_index is None or image_features is None:
            if labels.shape[1] == merged_seq_len:
                return labels
            pad = torch.full(
                (labels.shape[0], max(0, merged_seq_len - labels.shape[1])),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels, pad], dim=1)[:, :merged_seq_len]

        num_image_tokens = int(image_features.shape[1])
        expanded_rows = []
        for row_idx in range(labels.shape[0]):
            row_labels = []
            for tok_idx in range(input_ids.shape[1]):
                if attention_mask is not None and int(attention_mask[row_idx, tok_idx].item()) == 0:
                    continue
                token_id = int(input_ids[row_idx, tok_idx].item())
                label_val = labels[row_idx, tok_idx]
                if token_id == int(image_token_index):
                    row_labels.extend([label_val.new_tensor(-100) for _ in range(num_image_tokens)])
                else:
                    row_labels.append(label_val)
            if len(row_labels) == 0:
                row_tensor = torch.empty((0,), dtype=labels.dtype, device=labels.device)
            else:
                row_tensor = torch.stack(row_labels)
            if row_tensor.shape[0] < merged_seq_len:
                pad = torch.full(
                    (merged_seq_len - row_tensor.shape[0],),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                row_tensor = torch.cat([row_tensor, pad], dim=0)
            else:
                row_tensor = row_tensor[:merged_seq_len]
            expanded_rows.append(row_tensor)
        return torch.stack(expanded_rows, dim=0)

    def _extract_llava_image_features(self, model, pixel_values, image_sizes=None):
        get_image_features = getattr(model, "get_image_features", None)
        if get_image_features is not None:
            base_kwargs = {
                "pixel_values": pixel_values,
                "vision_feature_layer": getattr(model.config, "vision_feature_layer", None),
                "vision_feature_select_strategy": getattr(
                    model.config, "vision_feature_select_strategy", None
                ),
            }
            if image_sizes is not None:
                base_kwargs["image_sizes"] = image_sizes
            candidate_kwargs = [
                base_kwargs,
                {k: v for k, v in base_kwargs.items() if k != "image_sizes"},
                {"pixel_values": pixel_values},
            ]
            for kwargs in candidate_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    return get_image_features(**kwargs)
                except TypeError:
                    continue

        vision_tower = getattr(model, "vision_tower", None)
        projector = getattr(model, "multi_modal_projector", None)
        if vision_tower is None or projector is None:
            raise RuntimeError("LLaVA image feature extraction helpers are unavailable.")

        try:
            vision_out = vision_tower(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        except TypeError:
            vision_out = vision_tower(pixel_values, output_hidden_states=True)
        hidden_states = getattr(vision_out, "hidden_states", None)
        if hidden_states is None and isinstance(vision_out, (tuple, list)) and len(vision_out) > 2:
            hidden_states = vision_out[2]
        if hidden_states is None:
            raise RuntimeError("Vision tower did not return hidden states.")

        select_layer = int(getattr(model.config, "vision_feature_layer", -2))
        image_features = hidden_states[select_layer]
        if getattr(model.config, "vision_feature_select_strategy", "default") == "default":
            image_features = image_features[:, 1:]
        return projector(image_features)

    def _merge_llava_inputs_with_images(
        self,
        model,
        input_ids,
        attention_mask,
        inputs_embeds,
        image_features,
    ):
        merge_fn = getattr(model, "_merge_input_ids_with_image_features", None)
        if merge_fn is not None:
            merge_calls = [
                (image_features, inputs_embeds, input_ids, attention_mask, None),
                (image_features, inputs_embeds, input_ids, attention_mask),
            ]
            for merge_args in merge_calls:
                try:
                    merged = merge_fn(*merge_args)
                except TypeError:
                    continue
                if isinstance(merged, tuple) and len(merged) >= 2:
                    return merged[0], merged[1]

        image_token_index = getattr(model.config, "image_token_index", None)
        if image_token_index is None:
            return inputs_embeds, attention_mask

        special_image_mask = input_ids == image_token_index
        expected = special_image_mask.sum().item()
        actual = int(image_features.shape[0] * image_features.shape[1])
        if expected != actual:
            raise RuntimeError(
                f"Image token count mismatch: expected={expected}, actual={actual}."
            )
        flat_image_features = image_features.reshape(-1, image_features.shape[-1]).to(inputs_embeds.dtype)
        merged_embeds = inputs_embeds.clone()
        merged_embeds[special_image_mask] = flat_image_features
        return merged_embeds, attention_mask

    def _build_llava_text_image_embeddings(self, prompts, images, device, dtype):
        model = self._llava_model
        processor_batch = self._processor_call(prompts, images=images, padding=True, truncation=True)
        processor_batch = self._move_processor_batch_to_device(processor_batch, device, dtype)
        input_ids = processor_batch["input_ids"]
        attention_mask = processor_batch["attention_mask"]
        inputs_embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype)

        pixel_values = processor_batch.get("pixel_values", None)
        if pixel_values is None:
            return inputs_embeds, attention_mask, processor_batch

        image_sizes = processor_batch.get("image_sizes", None)
        need_grad = self.training and self._llava_vision_requires_grad()
        with torch.set_grad_enabled(need_grad):
            image_features = self._extract_llava_image_features(
                model=model,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
        if not need_grad:
            image_features = image_features.detach()
        image_features = self._pool_llava_image_features(image_features)
        image_features = image_features.to(device=device, dtype=dtype)
        inputs_embeds, attention_mask = self._merge_llava_inputs_with_images(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        return inputs_embeds, attention_mask, processor_batch

    def _build_joint_drive_qa_text(self, sample_group, ego_instruction=None):
        prompt_lines = [self.llava_train_prompt.strip()]
        # doScenes natural-language instruction → Alpaca-style "Instruction:".
        # This sits at the prompt front, BEFORE the BEV/agent/map/ego prefix
        # tokens (the cat order is also flipped to [text, prefix] so ego_token
        # positions can attend to the instruction under causal masking).
        if isinstance(ego_instruction, str) and ego_instruction.strip():
            prompt_lines.insert(0, f'Instruction: {ego_instruction.strip()}')
        answer_lines = []
        if isinstance(sample_group, (list, tuple)) and len(sample_group) > 0:
            prompt_lines.append("Answer the following driving questions:")
            for sample in sample_group:
                if not isinstance(sample, dict):
                    continue
                category = sample.get("category", "Question")
                question = sample.get("question", None)
                answer = sample.get("answer", None)
                if not isinstance(question, str):
                    continue
                category = str(category).strip() or "Question"
                prompt_lines.append(f"{category}: {question.strip()}")
                if isinstance(answer, str):
                    answer_lines.append(f"{category}: {answer.strip()}")

        prompt_body = "\n".join(line for line in prompt_lines if isinstance(line, str) and line.strip())
        answer_body = None
        if len(answer_lines) > 0:
            answer_body = "Answers:\n" + "\n".join(answer_lines)
        return prompt_body, answer_body

    def _append_eos(self, text: str) -> str:
        text = text.rstrip()
        if self._llava_tokenizer is None or self._llava_tokenizer.eos_token is None:
            return text
        if text.endswith(self._llava_tokenizer.eos_token):
            return text
        return text + self._llava_tokenizer.eos_token

    @staticmethod
    def _pool_token_span(hidden, token_span):
        if hidden is None or hidden.numel() == 0:
            return None
        if token_span is None:
            return hidden.mean(dim=1)
        start, end = int(token_span[0]), int(token_span[1])
        start = max(0, min(start, hidden.shape[1]))
        end = max(start + 1, min(end, hidden.shape[1]))
        span_hidden = hidden[:, start:end]
        if span_hidden.shape[1] == 0:
            return hidden.mean(dim=1)
        return span_hidden.mean(dim=1)

    def _hidden_to_plan_prediction(self, hidden, batch_size, ego_token_range=None):
        if hidden is None:
            return None
        if not torch.isfinite(hidden).all():
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)
        pooled = self._pool_token_span(hidden, ego_token_range)
        if pooled is None:
            return None
        head = self.pts_bbox_head
        ego_mode = int(getattr(head, "ego_fut_mode", 3))
        fut_ts = int(getattr(head, "fut_ts", self.fut_ts))
        # v7 soft routing: head emits 1 trajectory regardless of cmd
        soft_routing = bool(getattr(head, "enable_soft_cmd_routing", False))
        out_mode = 1 if soft_routing else ego_mode
        return self._llava_plan_head(pooled.float()).reshape(batch_size, out_mode, fut_ts, 2)

    @staticmethod
    def _extract_generation_prompt_hidden(generate_out):
        hidden_steps = getattr(generate_out, "hidden_states", None)
        if hidden_steps is None or len(hidden_steps) == 0:
            return None
        first_step = hidden_steps[0]
        if isinstance(first_step, (tuple, list)) and len(first_step) > 0:
            return first_step[-1]
        if torch.is_tensor(first_step):
            return first_step
        return None

    def _decode_generation_response(self, generate_out, prompt_length):
        sequences = getattr(generate_out, "sequences", None)
        if sequences is None or self._llava_tokenizer is None:
            return None
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        start_idx = prompt_length if sequences.shape[1] > prompt_length else 0
        response_ids = sequences[:, start_idx:]
        response = self._llava_tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        if not isinstance(response, list) or len(response) == 0:
            return None
        response = response[0].strip()
        return response or None

    def _forward_llava_joint_train(self, outs, llava_mosaic_img=None, drive_qa_samples=None,
                                   ego_instructions=None):
        zero = outs["ego_fut_preds"].sum() * 0.0
        if not self._lazy_init_llava():
            self._log_warning_once(
                "llava_joint_train_disabled",
                f"[VADLLaVA] Joint train skipped. init_state={self._llava_init_state}, "
                f"runtime_error={self._llava_runtime_error}",
            )
            return None, zero

        device, dtype = self._get_model_device_dtype()
        batch_size = outs["ego_fut_preds"].shape[0]
        images = self._resolve_llava_images(llava_mosaic_img)
        image_features = self._prepare_llava_image_features(
            device=device,
            dtype=dtype,
            images=images,
        )
        use_image = image_features is not None

        prefix_embeds, prefix_token_ranges = self._encode_training_prefix(
            outs=outs,
            device=device,
            dtype=dtype,
            return_token_ranges=True,
        )
        if prefix_embeds is None:
            return None, zero

        prompts = []
        full_texts = []
        qa_row_mask = []

        for sample_idx in range(batch_size):
            sample_group = None
            if self.drive_qa and drive_qa_samples is not None and sample_idx < len(drive_qa_samples):
                sample_group = drive_qa_samples[sample_idx]
            ego_inst = None
            if ego_instructions is not None and sample_idx < len(ego_instructions):
                ego_inst = ego_instructions[sample_idx]
            prompt_body, answer_body = self._build_joint_drive_qa_text(
                sample_group, ego_instruction=ego_inst)
            prompt = self._format_llava_prompt(prompt_body, include_image=use_image)
            prompts.append(prompt)
            if isinstance(answer_body, str) and answer_body.strip():
                full_texts.append(self._append_eos(f"{prompt}\n{answer_body.strip()}"))
                qa_row_mask.append(True)
            else:
                full_texts.append(prompt)
                qa_row_mask.append(False)

        if len(prompts) == 0:
            return None, zero
        _, prompt_attention_mask, _ = self._tokenize_llava_prompts(
            prompts=prompts,
            device=device,
            dtype=dtype,
        )
        full_input_ids, full_attention_mask_text, full_inputs_embeds = self._tokenize_llava_prompts(
            prompts=full_texts,
            device=device,
            dtype=dtype,
        )

        full_attention_mask = full_attention_mask_text
        if use_image:
            full_inputs_embeds, full_attention_mask = self._merge_llava_inputs_with_images(
                model=self._llava_model,
                input_ids=full_input_ids,
                attention_mask=full_attention_mask_text,
                inputs_embeds=full_inputs_embeds,
                image_features=image_features,
            )

        # Architecture fix (option A): place the BEV/agent/map/ego prefix AFTER
        # the text so the ego token can attend to "Instruction: ..." under
        # Llama's causal mask.  All token ranges shift by text length.
        text_len = full_attention_mask.shape[1]
        prefix_mask = torch.ones(
            (prefix_embeds.shape[0], prefix_embeds.shape[1]),
            dtype=full_attention_mask.dtype,
            device=full_attention_mask.device,
        )
        attention_mask = torch.cat([full_attention_mask, prefix_mask], dim=1)
        inputs_embeds = torch.cat([full_inputs_embeds, prefix_embeds], dim=1)
        if prefix_token_ranges:
            prefix_token_ranges = {
                k: (s + text_len, e + text_len)
                for k, (s, e) in prefix_token_ranges.items()
            }

        labels = full_input_ids.clone()
        labels[full_attention_mask_text == 0] = -100
        prompt_lens = prompt_attention_mask.sum(dim=1)
        for row_idx, has_qa in enumerate(qa_row_mask):
            if not has_qa:
                labels[row_idx, :] = -100
                continue
            labels[row_idx, : int(prompt_lens[row_idx].item())] = -100
        labels = self._expand_labels_with_image_tokens(
            input_ids=full_input_ids,
            labels=labels,
            image_features=image_features,
            merged_seq_len=full_attention_mask.shape[1],
            attention_mask=full_attention_mask_text,
        )
        prefix_ignore = torch.full(
            (labels.shape[0], prefix_embeds.shape[1]),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        # labels order matches inputs_embeds: [text_labels, prefix_ignore]
        labels = torch.cat([labels, prefix_ignore], dim=1)
        hidden = self._forward_lm_last_hidden(
            lm_model=self._llava_model.language_model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if not torch.isfinite(hidden).all():
            finite_ratio = torch.isfinite(hidden).float().mean().item()
            self._log_warning_once(
                "llava_joint_hidden_non_finite",
                f"[VADLLaVA] Joint hidden has non-finite values. finite_ratio={finite_ratio:.6f}. Apply nan_to_num.",
            )
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)

        planning_preds = self._hidden_to_plan_prediction(
            hidden=hidden,
            batch_size=batch_size,
            ego_token_range=prefix_token_ranges.get("ego"),
        )

        qa_loss = zero
        qa_rows = torch.as_tensor(qa_row_mask, dtype=torch.bool, device=hidden.device)
        if qa_rows.any():
            qa_loss = self._compute_causal_lm_loss_from_hidden(
                lm_head=self._llava_model.get_output_embeddings(),
                hidden=hidden[qa_rows],
                labels=labels[qa_rows],
            )
            if (not torch.isfinite(qa_loss)) or (qa_loss < 0) or (qa_loss > 10):
                print('qa_loss', qa_loss)
            if qa_loss is None or not torch.isfinite(qa_loss):
                self._log_warning_once(
                    "llava_joint_qa_non_finite",
                    "[VADLLaVA] Joint QA loss is non-finite. Skip QA loss this iter.",
                )
                qa_loss = zero
            else:
                qa_loss = qa_loss * float(self.llava_qa_loss_weight)

        return planning_preds, qa_loss

    def _prepare_train_bev_tokens(self, outs):
        bev = outs.get("bev_embed", None)
        bev = bev.permute(1, 0, 2)

        if bev.shape[1] > self.llava_bev_topk:
            idx = torch.linspace(0, bev.shape[1] - 1, steps=self.llava_bev_topk, device=bev.device).long()
            bev = bev[:, idx, :]
        return bev

    def _prepare_train_agent_tokens(self, outs):
        agent_bbox = outs["all_bbox_preds"][-1]
        agent_cls_logits = outs["all_cls_scores"][-1]
        agent_cls = agent_cls_logits.sigmoid()
        agent_score = agent_cls.max(dim=-1)[0].unsqueeze(-1)
        tokens = torch.cat([agent_bbox, agent_score], dim=-1)
        if tokens.shape[1] > self.llava_max_agents:
            score = agent_score.squeeze(-1)
            topk = torch.topk(score, k=self.llava_max_agents, dim=1).indices
            batch_idx = torch.arange(tokens.shape[0]).unsqueeze(-1).expand_as(topk)
            tokens = tokens[batch_idx, topk]
        return tokens

    def _prepare_train_map_tokens(self, outs):
        map_pts = outs["map_all_pts_preds"][-1]
        map_cls_logits = outs["map_all_cls_scores"][-1]
        map_cls = map_cls_logits.sigmoid()
        map_score = map_cls.max(dim=-1)[0].unsqueeze(-1)
        tokens = torch.cat([map_pts.flatten(start_dim=2), map_score], dim=-1)
        if tokens.shape[1] > self.llava_max_map_vectors:
            score = map_score.squeeze(-1)
            topk = torch.topk(score, k=self.llava_max_map_vectors, dim=1).indices
            batch_idx = torch.arange(tokens.shape[0]).unsqueeze(-1).expand_as(topk)
            tokens = tokens[batch_idx, topk]
        return tokens

    def _encode_prefix_tokens(
        self,
        token_sets,
        device,
        dtype,
        sanitize=False,
        return_token_ranges=False,
    ):
        if self._llava_hidden_size is None:
            return (None, {}) if return_token_ranges else None

        prefix_chunks = []
        token_ranges = {}
        token_offset = 0
        for token_type in ["bev", "agent", "map", "ego"]:
            raw_tokens = token_sets.get(token_type, None)
            if raw_tokens is None:
                continue
            if sanitize:
                raw_tokens = self._sanitize_tensor(
                    raw_tokens,
                    f"prefix_raw_{token_type}",
                    clamp_val=1e3,
                )
            proj_tokens = self._project_tokens(
                token_type=token_type,
                tokens=raw_tokens,
                hidden_size=self._llava_hidden_size,
                device=device,
                dtype=torch.float32,
            )
            if proj_tokens is None:
                continue

            if self.llava_use_planning_adapter:
                adapter = self._planning_adapters[token_type]
                adapted = adapter(proj_tokens)
            else:
                adapted = proj_tokens
            if token_type in self._planning_type_bias:
                type_bias = self._planning_type_bias[token_type]
                adapted = adapted + type_bias
            if adapted.requires_grad:
                adapted.register_hook(
                        lambda grad: torch.clamp(
                            torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0),
                            min=-1.0, max=1.0
                        )
                )
            if sanitize:
                adapted = self._sanitize_tensor(
                    adapted,
                    f"prefix_proj_{token_type}",
                    clamp_val=10.0,
                )
            adapted = adapted.to(dtype=dtype)
            prefix_chunks.append(adapted)
            token_ranges[token_type] = (token_offset, token_offset + adapted.shape[1])
            token_offset += adapted.shape[1]

        if len(prefix_chunks) == 0:
            return (None, token_ranges) if return_token_ranges else None

        prefix_embeds = torch.cat(prefix_chunks, dim=1)
        if return_token_ranges:
            return prefix_embeds, token_ranges
        return prefix_embeds

    def _encode_training_prefix(self, outs, device, dtype, return_token_ranges=False):
        if not self._llava_train_modules_initialized:
            return (None, {}) if return_token_ranges else None

        token_sets = {
            "bev": self._prepare_train_bev_tokens(outs),
            "agent": self._prepare_train_agent_tokens(outs),
            "map": self._prepare_train_map_tokens(outs),
            "ego": outs['ego_feats'],
        }
        return self._encode_prefix_tokens(
            token_sets=token_sets,
            device=device,
            dtype=dtype,
            sanitize=True,
            return_token_ranges=return_token_ranges,
        )

    def _forward_llava_plan_train(self, outs, llava_mosaic_img=None):
        model = self._llava_model
        lm_model = self._llava_model.language_model
        device, dtype = self._get_model_device_dtype()
        batch_size = outs["ego_fut_preds"].shape[0]
        images = self._resolve_llava_images(llava_mosaic_img)
        prompts = [
            self._format_llava_prompt(self.llava_train_prompt, include_image=images is not None)
            for _ in range(batch_size)
        ]
        if images is not None:
            text_embeds, attention_mask, _ = self._build_llava_text_image_embeddings(
                prompts=prompts,
                images=images,
                device=device,
                dtype=dtype,
            )
        else:
            tokenizer = self._llava_tokenizer
            text_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.llava_text_max_length) if self.llava_text_max_length is not None else None,
            )
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            text_embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype)

        prefix_embeds, prefix_token_ranges = self._encode_training_prefix(
            outs=outs,
            device=device,
            dtype=dtype,
            return_token_ranges=True,
        )
        # Architecture fix (option A): put the BEV/agent/map/ego prefix AFTER
        # the text so the ego token can attend to the "Instruction: ..." line
        # under Llama's causal mask.  Token ranges shift by text length.
        text_len = text_embeds.shape[1]
        prefix_mask = torch.ones(
            (attention_mask.shape[0], prefix_embeds.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)
        inputs_embeds = torch.cat([text_embeds, prefix_embeds], dim=1)
        if prefix_token_ranges:
            prefix_token_ranges = {
                k: (s + text_len, e + text_len)
                for k, (s, e) in prefix_token_ranges.items()
            }

        pred = self._decode_llava_hidden_to_plan(
            lm_model=lm_model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            batch_size=batch_size,
            device=device,
            ego_token_range=prefix_token_ranges.get("ego"),
        )
        return pred, None

    def _decode_llava_hidden_to_plan(
        self,
        lm_model,
        inputs_embeds,
        attention_mask,
        batch_size,
        device,
        ego_token_range=None,
    ):
        hidden = self._forward_lm_last_hidden(
            lm_model=lm_model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # bev(8) agent(8) map(8) ego(8) text(14) total 46: B, 46, 4096

        if not torch.isfinite(hidden).all():
            finite_ratio = torch.isfinite(hidden).float().mean().item()
            msg = f"[VADLLaVA] llm hidden has non-finite values. finite_ratio={finite_ratio:.6f}. Apply nan_to_num."
            self._logger.warning(msg)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)

        return self._hidden_to_plan_prediction(
            hidden=hidden,
            batch_size=batch_size,
            ego_token_range=ego_token_range,
        )

    def _pick_cmd_idx(self, ego_fut_cmd: Optional[torch.Tensor]) -> int:
        # v7 soft routing: head emits a single trajectory (mode dim = 1),
        # so cmd-as-mode-selector is degenerate — always pick index 0.
        if bool(getattr(self.pts_bbox_head, "enable_soft_cmd_routing", False)):
            return 0
        if ego_fut_cmd is None:
            return 0
        cmd = ego_fut_cmd
        if not torch.is_tensor(cmd):
            cmd = torch.as_tensor(cmd)
        cmd = cmd.detach().cpu()
        while cmd.dim() > 1:
            cmd = cmd[0]
        nonzero = torch.nonzero(cmd > 0)
        if nonzero.shape[0] == 0:
            return 0
        return int(nonzero[0, 0].item())

    def _safe_topk(self, scores, topk):
        if scores.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=scores.device)
        topk = max(0, min(int(topk), scores.shape[0]))
        if topk == 0:
            return torch.zeros((0,), dtype=torch.long, device=scores.device)
        return torch.argsort(scores, descending=True)[:topk]

    def _normalize_bev_tokens(self, bev_embed, device):
        if bev_embed is None or not torch.is_tensor(bev_embed):
            return None
        bev = bev_embed.detach().to(device)
        if bev.dim() == 2:
            bev = bev.unsqueeze(0)
        elif bev.dim() == 3:
            if bev.shape[1] == 1:
                bev = bev.permute(1, 0, 2).contiguous()
            elif bev.shape[0] != 1:
                bev = bev[:1]
        else:
            return None
        if bev.shape[1] > self.llava_bev_topk:
            indices = torch.linspace(
                0,
                bev.shape[1] - 1,
                steps=self.llava_bev_topk,
                device=bev.device,
            ).long()
            bev = bev[:, indices, :]
        return bev

    def _build_agent_tokens(self, bbox_result, device):
        scores = bbox_result["scores_3d"].detach().to(device)
        labels = bbox_result["labels_3d"].detach().to(device)
        boxes = bbox_result["boxes_3d"].tensor.detach().to(device)
        if scores.numel() == 0:
            return None
        idx = self._safe_topk(scores, self.llava_max_agents)
        if idx.numel() == 0:
            return None
        token = torch.cat(
            [
                boxes[idx, :9],
                scores[idx, None],
                (labels[idx, None] / 10.0),
            ],
            dim=-1,
        )
        return token.unsqueeze(0)

    def _build_map_tokens(self, bbox_result, device):
        scores = bbox_result["map_scores_3d"].detach().to(device)
        points = bbox_result["map_pts_3d"].detach().to(device)
        if scores.numel() == 0:
            return None
        idx = self._safe_topk(scores, self.llava_max_map_vectors)
        if idx.numel() == 0:
            return None

        selected_pts = points[idx]
        token = torch.cat(
            [
                selected_pts.flatten(start_dim=1),
                scores[idx, None],
            ],
            dim=-1,
        )
        return token.unsqueeze(0)

    def _build_ego_tokens(self, bbox_result, device):
        ego_feats = bbox_result.get("ego_feats", None)
        if ego_feats is None or not torch.is_tensor(ego_feats):
            return None
        ego_feats = ego_feats.detach().to(device)
        if ego_feats.dim() == 1:
            ego_feats = ego_feats[None, None, :]
        elif ego_feats.dim() == 2:
            ego_feats = ego_feats[:1].unsqueeze(0)
        elif ego_feats.dim() == 3:
            ego_feats = ego_feats[:1, :1, :]
        else:
            return None
        return ego_feats

    def _build_planning_token_sets(
        self,
        bev_embed,
        bbox_result,
        device,
    ):
        return {
            "bev": self._normalize_bev_tokens(bev_embed, device),
            "agent": self._build_agent_tokens(bbox_result, device),
            "map": self._build_map_tokens(bbox_result, device),
            "ego": self._build_ego_tokens(bbox_result, device),
        }

    def _build_projector(self, in_features, out_features):
        proj = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )
        nn.init.normal_(proj[-1].weight, std=0.02)
        nn.init.constant_(proj[-1].bias, 0.0)
        return proj

    def _project_tokens(self, token_type, tokens, hidden_size, device, dtype):
        if tokens is None or tokens.numel() == 0:
            return None

        if not self.llava_use_projector:
            planning_dims = self._planning_context_dims()
            target_dim = planning_dims[token_type]
            out = self._match_last_dim(
                tokens.to(device=device, dtype=dtype),
                target_dim,
            )
            return out

        in_features = int(tokens.shape[-1])
        if token_type not in self._token_projectors:
            self._token_projectors[token_type] = self._build_projector(
                in_features,
                hidden_size,
            )
        projector = self._token_projectors[token_type]
        first_linear = projector[0]
        if int(first_linear.in_features) != in_features:
            self._token_projectors[token_type] = self._build_projector(in_features, hidden_size)
            projector = self._token_projectors[token_type]
        projector = projector.to(device=device, dtype=torch.float32)
        out = projector(tokens.to(device=device, dtype=torch.float32))
        return out.to(dtype=dtype)

    def _encode_planning_prefix(
        self,
        bev_embed,
        bbox_result,
        device,
        dtype,
        return_token_ranges=False,
    ):
        token_sets = self._build_planning_token_sets(
            bev_embed=bev_embed,
            bbox_result=bbox_result,
            device=device,
        )
        return self._encode_prefix_tokens(
            token_sets=token_sets,
            device=device,
            dtype=dtype,
            sanitize=False,
            return_token_ranges=return_token_ranges,
        )

    def _run_llava(self, bev_embed, bbox_result, cmd_idx, llava_mosaic_img=None,
                   drive_qa_samples=None, ego_instruction=None):
        if not self._lazy_init_llava():
            return {
                "status": "disabled_or_init_failed",
                "error": self._llava_runtime_error,
                "response": None,
                "waypoints": None,
            }

        try:
            tokenizer = self._llava_tokenizer
            model = self._llava_model
            lm_model = self._llava_model.language_model
            model_device, model_dtype = self._get_model_device_dtype()
            if not self._llava_train_modules_initialized or self._llava_plan_head is None:
                return {
                    "status": "runtime_failed",
                    "error": "llava planning head is not initialized",
                    "response": None,
                    "waypoints": None,
                }

            images = self._resolve_llava_images(llava_mosaic_img)
            prompt_body, _ = self._build_joint_drive_qa_text(
                drive_qa_samples, ego_instruction=ego_instruction)
            prompt = self._format_llava_prompt(prompt_body, include_image=images is not None)
            if images is not None:
                text_embeds, attention_mask, _ = self._build_llava_text_image_embeddings(
                    prompts=[prompt],
                    images=images,
                    device=model_device,
                    dtype=model_dtype,
                )
            else:
                model_inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(self.llava_text_max_length) if self.llava_text_max_length is not None else None,
                )
                input_ids = model_inputs["input_ids"].to(model_device)
                attention_mask = model_inputs["attention_mask"].to(model_device)
                text_embeds = model.get_input_embeddings()(input_ids).to(dtype=model_dtype)

            prefix_embeds, prefix_token_ranges = self._encode_planning_prefix(
                bev_embed=bev_embed,
                bbox_result=bbox_result,
                device=model_device,
                dtype=model_dtype,
                return_token_ranges=True,
            )
            if prefix_embeds is not None and prefix_embeds.shape[1] > 0:
                # Architecture fix (option A): put prefix AFTER text so the
                # ego token can attend to the instruction text.  Shift all
                # prefix token ranges by text length.
                text_len = text_embeds.shape[1]
                prefix_mask = torch.ones(
                    (attention_mask.shape[0], prefix_embeds.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)
                inputs_embeds = torch.cat([text_embeds, prefix_embeds], dim=1)
                if prefix_token_ranges:
                    prefix_token_ranges = {
                        k: (s + text_len, e + text_len)
                        for k, (s, e) in prefix_token_ranges.items()
                    }
            else:
                inputs_embeds = text_embeds
            with torch.no_grad():
                generate_out = lm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=1,  # plan_head only uses hidden_states[0] (prompt-end), so 1 is enough
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_hidden = self._extract_generation_prompt_hidden(generate_out)
            pred = self._hidden_to_plan_prediction(
                hidden=prompt_hidden,
                batch_size=1,
                ego_token_range=prefix_token_ranges.get("ego"),
            )
            response = self._decode_generation_response(generate_out, attention_mask.shape[1])
            cmd_idx = max(0, min(cmd_idx, pred.shape[1] - 1))
            waypoints = pred[0, cmd_idx].detach().cpu()
            return {
                "status": "ok",
                "error": None,
                "response": response,
                "waypoints": waypoints,
                "all_modes": pred[0].detach().cpu(),
            }
        except Exception as exc:
            print(exc)
            return {
                "status": "runtime_failed",
                "error": str(exc),
                "response": None,
                "waypoints": None,
            }

    def _attach_llava_result(self, bbox_result, llava_out, cmd_idx):
        bbox_result["llava_status"] = llava_out["status"]
        bbox_result["llava_error"] = llava_out["error"]
        bbox_result["llava_response"] = llava_out["response"]
        # Preserve original vision planner output for downstream analysis/visualization.
        if "ego_fut_preds" in bbox_result and "ego_fut_preds_vision" not in bbox_result:
            bbox_result["ego_fut_preds_vision"] = bbox_result["ego_fut_preds"].clone()
        if llava_out["waypoints"] is None:
            bbox_result["llava_waypoints"] = None
            return

        llava_waypoints = llava_out["waypoints"].cpu()
        bbox_result["llava_waypoints"] = llava_waypoints
        if "all_modes" in llava_out and llava_out["all_modes"] is not None:
            bbox_result["llava_waypoints_all_modes"] = llava_out["all_modes"]

        ego_fut_preds = bbox_result["ego_fut_preds"].clone()
        cmd_idx = max(0, min(cmd_idx, ego_fut_preds.shape[0] - 1))
        ego_fut_preds[cmd_idx] = llava_waypoints.to(ego_fut_preds.dtype)
        bbox_result["ego_fut_preds_llava"] = ego_fut_preds
        if self.llava_replace_ego_fut_preds:
            bbox_result["ego_fut_preds"] = ego_fut_preds

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        map_gt_bboxes_ignore=None,
        prev_bev=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_masks=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        llava_mosaic_img=None,
        drive_qa_samples=None,
    ):
        outs = self.pts_bbox_head(
            pts_feats,
            img_metas,
            prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_lcf_feat=ego_lcf_feat,
            ego_fut_cmd=ego_fut_cmd,
        )

        # Stage C: inject text-conditioned additive delta on ego_fut_preds
        # BEFORE loss is computed, so gradients flow back to TextDeltaPlanner
        # (LoRA + Linear + MLP + alpha). No-op when planner disabled.
        if self.text_delta_planner_enabled and self._text_delta_planner is not None:
            ego_instructions_local = [
                self._build_instruction_text(m)
                for m in (img_metas if isinstance(img_metas, (list, tuple)) else [])
            ]
            self._apply_text_delta(outs, ego_instructions_local)

        loss_inputs = [
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            outs,
            ego_fut_trajs,
            ego_fut_masks,
            ego_fut_cmd,
            gt_attr_labels,
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        # Level-2 FDE recipe: auxiliary endpoint loss directly supervises the
        # language-predicted 6s goal with GT t=6s position, weighted by
        # ego_fut_cmd one-hot (same mode-pick philosophy as plan_reg).
        # ALWAYS register the loss key (with 0 value when no endpoint present)
        # to keep DDP's per-rank loss-key set consistent.
        if self.text_delta_use_endpoint_head and self.text_delta_endpoint_loss_weight > 0:
            zero_ep_ref = outs["ego_fut_preds"].sum() * 0.0
            endpoint = outs.get('text_delta_endpoint', None)
            if endpoint is not None and ego_fut_trajs is not None and ego_fut_cmd is not None:
                efts = ego_fut_trajs
                if efts.dim() == 3 and efts.shape[-1] != 2:
                    B_, _, TT2 = efts.shape
                    gt_xy = efts.reshape(B_, TT2 // 2, 2)
                elif efts.dim() == 4:
                    gt_xy = efts.squeeze(1)
                else:
                    gt_xy = efts
                gt_endpoint = gt_xy.cumsum(dim=-2)[:, -1, :]
                endpoint = endpoint.to(gt_endpoint.dtype)
                cmd = ego_fut_cmd
                while cmd.dim() > 2:
                    cmd = cmd.squeeze(1)
                cmd = cmd.to(endpoint.dtype)
                if self.text_delta_endpoint_loss_type == 'smooth_l1':
                    # Per-mode Huber loss averaged over xy dims
                    import torch.nn.functional as F
                    err = F.smooth_l1_loss(
                        endpoint, gt_endpoint.unsqueeze(1).expand_as(endpoint),
                        reduction='none', beta=1.0).mean(dim=-1)  # [B, M]
                else:
                    err = (endpoint - gt_endpoint.unsqueeze(1)).norm(dim=-1)
                loss_ep = (err * cmd).sum(-1).mean()
                losses['loss_text_delta_endpoint'] = (
                    self.text_delta_endpoint_loss_weight * loss_ep + zero_ep_ref)
            else:
                # All-empty batch on this rank — keep DDP loss-key set consistent.
                losses['loss_text_delta_endpoint'] = zero_ep_ref

        # Speed-class auxiliary CE loss (always register key for DDP consistency)
        if self.text_delta_use_speed_class_head and \
                self.text_delta_speed_class_loss_weight > 0:
            speed_logits = outs.get('text_delta_speed_logits', None)
            keep_idx = outs.get('text_delta_speed_keep_idx', None)
            if speed_logits is not None and keep_idx is not None and \
                    ego_fut_trajs is not None and speed_logits.numel() > 0:
                import torch.nn.functional as F
                efts = ego_fut_trajs
                if efts.dim() == 3 and efts.shape[-1] != 2:
                    B_, _, TT2 = efts.shape
                    gt_xy = efts.reshape(B_, TT2 // 2, 2)
                elif efts.dim() == 4:
                    gt_xy = efts.squeeze(1)
                else:
                    gt_xy = efts
                step_norms = gt_xy.norm(dim=-1)
                total_dist = step_norms.sum(dim=-1)
                avg_speed = total_dist / 6.0
                bin_edges = torch.tensor(
                    self.text_delta_speed_bin_edges,
                    device=avg_speed.device, dtype=avg_speed.dtype)
                gt_class_full = torch.bucketize(avg_speed, bin_edges)
                gt_class_kept = gt_class_full[keep_idx]
                loss_sp = F.cross_entropy(
                    speed_logits.float(), gt_class_kept.long())
                # No zero_sp_ref — keeps loss path orthogonal to plan_reg's
                # ego_fut_preds graph, avoiding "backward through graph twice"
                losses['loss_text_delta_speed'] = (
                    self.text_delta_speed_class_loss_weight * loss_sp)
            else:
                # Use planner param to keep key registered (single-GPU safe)
                losses['loss_text_delta_speed'] = (
                    0.0 * self._text_delta_planner.speed_class_head.weight.sum())

        zero_ref = outs["ego_fut_preds"].sum() * 0.0
        # Stage C "planner-only" mode: skip original LLaVA Q-Former + generate path.
        if self.text_delta_planner_only:
            return losses
        if not self.llava_enabled and not self.drive_qa:
            return losses

        # doScenes instruction (per-sample) is carried in img_metas thanks to
        # the LoadDoScenesInstruction pipeline transform.
        ego_instructions = None
        if isinstance(img_metas, (list, tuple)) and len(img_metas) > 0:
            ego_instructions = [
                (m.get('ego_instruction', '') if isinstance(m, dict) else '')
                for m in img_metas
            ]
        llava_ego_fut_preds, qa_loss = self._forward_llava_joint_train(
            outs=outs,
            llava_mosaic_img=llava_mosaic_img,
            drive_qa_samples=drive_qa_samples,
            ego_instructions=ego_instructions,
        )
        if self.drive_qa:
            losses["loss_llava_qa"] = qa_loss

        if not self.llava_enabled:
            return losses

        if ego_fut_trajs is None or ego_fut_masks is None or ego_fut_cmd is None:
            losses["loss_llava_plan_reg"] = zero_ref
            if self.llava_use_plan_constraint_loss:
                losses["loss_llava_plan_bound"] = zero_ref
                losses["loss_llava_plan_col"] = zero_ref
                losses["loss_llava_plan_dir"] = zero_ref
            return losses
        if llava_ego_fut_preds is None:
            self._log_warning_once(
                "llava_plan_train_disabled",
                f"[VADLLaVA] llava_ego_fut_preds is None. init_state={self._llava_init_state}, "
                f"runtime_error={self._llava_runtime_error}",
            )
            losses["loss_llava_plan_reg"] = zero_ref
            if self.llava_use_plan_constraint_loss:
                losses["loss_llava_plan_bound"] = zero_ref
                losses["loss_llava_plan_col"] = zero_ref
                losses["loss_llava_plan_dir"] = zero_ref
            return losses
        llava_ego_fut_preds = llava_ego_fut_preds.to(
            device=outs["ego_fut_preds"].device,
            dtype=outs["ego_fut_preds"].dtype,
        )
        if not torch.isfinite(llava_ego_fut_preds).all():
            self._log_warning_once(
                "llava_pred_non_finite",
                "[VADLLaVA] llava_ego_fut_preds has non-finite values. Skip llava loss this iter.",
            )
            losses["loss_llava_plan_reg"] = zero_ref
            if self.llava_use_plan_constraint_loss:
                losses["loss_llava_plan_bound"] = zero_ref
                losses["loss_llava_plan_col"] = zero_ref
                losses["loss_llava_plan_dir"] = zero_ref
            return losses

        ego_fut_gt = ego_fut_trajs.squeeze(1)
        ego_fut_masks_s = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd_s = ego_fut_cmd.squeeze(1).squeeze(1)

        if self.llava_use_plan_constraint_loss:
            traj_preds = outs["all_traj_preds"][-1]
            traj_cls_scores = outs["all_traj_cls_scores"][-1]
            map_all_pts = outs["map_all_pts_preds"][-1]
            map_all_cls_scores = outs["map_all_cls_scores"][-1]
            all_bbox_preds = outs["all_bbox_preds"][-1]
            all_cls_scores = outs["all_cls_scores"][-1]

            batch, num_agent = traj_preds.shape[:2]
            agent_fut_preds = traj_preds.view(
                batch,
                num_agent,
                self.pts_bbox_head.fut_mode,
                self.pts_bbox_head.fut_ts,
                2,
            )
            agent_fut_cls_preds = traj_cls_scores.view(
                batch,
                num_agent,
                self.pts_bbox_head.fut_mode,
            )
            llava_plan_dict = self.pts_bbox_head.loss_planning(
                llava_ego_fut_preds,
                ego_fut_gt,
                ego_fut_masks_s,
                ego_fut_cmd_s,
                map_all_pts,
                map_all_cls_scores.sigmoid(),
                all_bbox_preds[..., 0:2],
                agent_fut_preds,
                all_cls_scores.sigmoid(),
                agent_fut_cls_preds.sigmoid(),
            )
            losses["loss_llava_plan_reg"] = llava_plan_dict["loss_plan_reg"] * self.llava_plan_loss_weight
            losses["loss_llava_plan_bound"] = llava_plan_dict["loss_plan_bound"] * self.llava_plan_loss_weight
            losses["loss_llava_plan_col"] = llava_plan_dict["loss_plan_col"] * self.llava_plan_loss_weight
            losses["loss_llava_plan_dir"] = llava_plan_dict["loss_plan_dir"] * self.llava_plan_loss_weight
        else:
            ego_fut_gt_r = ego_fut_gt.unsqueeze(1).repeat(1, llava_ego_fut_preds.shape[1], 1, 1)
            soft_routing = bool(getattr(self.pts_bbox_head, "enable_soft_cmd_routing", False))
            if soft_routing:
                # single trajectory: weight by mask only, no cmd-mode mask
                llava_w = ego_fut_masks_s[:, None, :, None].repeat(
                    1, llava_ego_fut_preds.shape[1], 1, 2)
            else:
                llava_w = ego_fut_cmd_s[..., None, None] * ego_fut_masks_s[:, None, :, None]
                llava_w = llava_w.repeat(1, 1, 1, 2)
            llava_loss_reg = self.pts_bbox_head.loss_plan_reg(
                llava_ego_fut_preds,
                ego_fut_gt_r,
                llava_w,
            )
            if not torch.isfinite(llava_loss_reg):
                self._log_warning_once(
                    "llava_plan_reg_non_finite",
                    "[VADLLaVA] llava plan reg loss is non-finite. Skip llava loss this iter.",
                )
                losses["loss_llava_plan_reg"] = zero_ref
                return losses
            losses["loss_llava_plan_reg"] = llava_loss_reg * self.llava_plan_loss_weight

        return losses

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        map_gt_bboxes_3d=None,
        map_gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        map_gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_masks=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        llava_mosaic_img=None,
        drive_qa_samples=None,
    ):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        cur_img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        img_metas = [each[len_queue - 1] for each in img_metas]
        img_feats = self.extract_feat(img=cur_img, img_metas=img_metas)
        losses = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            map_gt_bboxes_ignore,
            prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_masks=ego_fut_masks,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            llava_mosaic_img=llava_mosaic_img,
            drive_qa_samples=drive_qa_samples,
        )
        return dict(losses)

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        llava_mosaic_img=None,
        drive_qa_samples=None,
    ):
        bev_embed, bbox_results, metric_dict = super().simple_test_pts(
            x=x,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            prev_bev=prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=start,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )

        # Stage C inference: apply text-delta residual to ego_fut_preds before
        # any downstream consumer (eval_doscenes_local reads bbox_result['ego_fut_preds']).
        if self.text_delta_planner_enabled and self._text_delta_planner is not None \
                and len(bbox_results) > 0:
            ego_instr_single = ['']
            if isinstance(img_metas, (list, tuple)) and len(img_metas) > 0 \
                    and isinstance(img_metas[0], dict):
                ego_instr_single = [self._build_instruction_text(img_metas[0])]
            self._apply_text_delta(bbox_results[0], ego_instr_single)

        if not self.llava_enabled or len(bbox_results) == 0 or self.text_delta_planner_only:
            return bev_embed, bbox_results, metric_dict

        cmd_idx = self._pick_cmd_idx(ego_fut_cmd)
        # doScenes instruction (single anchor sample at inference time)
        ego_instruction = None
        if isinstance(img_metas, (list, tuple)) and len(img_metas) > 0:
            anchor_meta = img_metas[0]
            if isinstance(anchor_meta, dict):
                ego_instruction = anchor_meta.get('ego_instruction', '') or None
        llava_out = self._run_llava(
            bev_embed=bev_embed,
            bbox_result=bbox_results[0],
            cmd_idx=cmd_idx,
            llava_mosaic_img=llava_mosaic_img,
            drive_qa_samples=drive_qa_samples[0] if isinstance(drive_qa_samples, (list, tuple)) and len(drive_qa_samples) > 0 else drive_qa_samples,
            ego_instruction=ego_instruction,
        )
        self._attach_llava_result(bbox_results[0], llava_out, cmd_idx)

        if (
            llava_out["waypoints"] is not None
            and ego_fut_trajs is not None
            and fut_valid_flag is not None
            and gt_bboxes_3d is not None
            and gt_attr_labels is not None
        ):
            with torch.no_grad():
                gt_ego_fut_trajs = ego_fut_trajs[0, 0].detach().cpu().cumsum(dim=-2)
                pred_ego_fut_trajs = llava_out["waypoints"].detach().cpu().cumsum(dim=-2)
                gt_bbox = gt_bboxes_3d[0][0]
                gt_attr = gt_attr_labels[0][0].to("cpu")
                valid_flag = bool(fut_valid_flag[0][0])
                llava_metrics = self.compute_planner_metric_stp3(
                    pred_ego_fut_trajs=pred_ego_fut_trajs[None],
                    gt_ego_fut_trajs=gt_ego_fut_trajs[None],
                    gt_agent_boxes=gt_bbox,
                    gt_agent_feats=gt_attr.unsqueeze(0),
                    fut_valid_flag=valid_flag,
                )
            for key, value in llava_metrics.items():
                metric_dict[f"llava_{key}"] = value

        return bev_embed, bbox_results, metric_dict

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        llava_mosaic_img=None,
        drive_qa_samples=None,
        **kwargs
    ):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            llava_mosaic_img=llava_mosaic_img,
            drive_qa_samples=drive_qa_samples,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict

        return new_prev_bev, bbox_list
