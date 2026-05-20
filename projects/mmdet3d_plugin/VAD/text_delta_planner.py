"""TextDeltaPlanner — language-conditioned additive delta on ego_fut_preds.

Architecture (Stage C):

    ego_instruction (str list, batch B)
        │ tokenize (max_length=64)
        ▼
    LLaMA language_model forward (frozen base + LoRA on q/v)
        │ hidden_states[-1]    [B, L, 4096]
        ▼
    masked mean-pool over real tokens (no PAD, no PREFIX)
        │
        ▼
    text_vec [B, 4096]
        │ Linear(4096 → text_proj_dim)
        ▼
    text_proj [B, text_proj_dim]
        │ concat ego_feats   [B, ego_feat_dim]
        ▼
    fused [B, text_proj_dim + ego_feat_dim]
        │ MLP(... → ego_fut_mode * fut_ts * 2)
        ▼
    delta [B, ego_fut_mode, fut_ts, 2]
        │ × alpha (Parameter, init=0)
        ▼
    α · delta    (added to ego_fut_preds elsewhere)

Properties:
  * α init = 0  →  initial output is ego_fut_preds unchanged (= v4_resume baseline).
  * MLP last-layer init small  →  delta starts ≈ 0 too (safety belt).
  * non-empty mask: samples with blank instruction get 0 contribution.
  * The LLaVA model is OWNED by VADLLaVA. We don't double-register it
    here — caller passes `language_model` into forward() at runtime.

Trainable params (~10-12 M total, depending on dims):
  * text_proj.weight/bias       ~1 M
  * mlp.*                       ~1-2 M
  * alpha                       1
  * LoRA adapters on LLaMA q/v  ~8.4 M  (owned by VADLLaVA, not here)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class TextDeltaPlanner(nn.Module):
    def __init__(
        self,
        llava_tokenizer,
        ego_feat_dim: Optional[int] = None,   # if None, lazy-init on first forward
        text_hidden_size: int = 4096,         # LLaMA-1.5-7b hidden
        ego_fut_mode: int = 3,
        fut_ts: int = 12,
        text_proj_dim: int = 256,
        mlp_hidden_dim: int = 512,
        text_max_length: int = 64,
        alpha_init: float = 0.0,
        mlp_last_init_std: float = 0.01,
    ):
        super().__init__()
        # Tokenizer is not a torch module; safe to hold as attr without registration.
        self._llava_tokenizer = llava_tokenizer

        self.text_hidden_size = int(text_hidden_size)
        self.ego_fut_mode = int(ego_fut_mode)
        self.fut_ts = int(fut_ts)
        self.text_proj_dim = int(text_proj_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.text_max_length = int(text_max_length)
        self.mlp_last_init_std = float(mlp_last_init_std)

        # Trainable: text projection
        self.text_proj = nn.Linear(self.text_hidden_size, self.text_proj_dim)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)

        # MLP gets lazy-init when ego_feat_dim is known (on first forward).
        self.ego_feat_dim = None
        self.mlp: Optional[nn.Sequential] = None
        if ego_feat_dim is not None:
            self._build_mlp(int(ego_feat_dim))

        # Gated alpha — init 0 so initial output is unconditional baseline.
        self.alpha = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float32))

    # ----------------------------- helpers -----------------------------

    def _build_mlp(self, ego_feat_dim: int) -> None:
        out_dim = self.ego_fut_mode * self.fut_ts * 2
        self.ego_feat_dim = int(ego_feat_dim)
        mlp = nn.Sequential(
            nn.Linear(self.text_proj_dim + self.ego_feat_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, out_dim),
        )
        # First-layer normal init, last-layer small init so delta ≈ 0.
        nn.init.normal_(mlp[0].weight, std=0.02)
        nn.init.zeros_(mlp[0].bias)
        nn.init.normal_(mlp[-1].weight, std=self.mlp_last_init_std)
        nn.init.zeros_(mlp[-1].bias)
        self.mlp = mlp

    @torch.no_grad()
    def _move_to(self, device, dtype):
        """Move newly lazy-built MLP to the same device/dtype as the rest."""
        if self.mlp is not None:
            self.mlp.to(device=device, dtype=dtype)

    # ----------------------------- text encode -----------------------------

    def encode_text(self, texts: List[str], language_model, device,
                    dtype) -> torch.Tensor:
        """Run text-only LLaMA forward and mean-pool over real tokens.

        Args:
          texts: list of B instruction strings (assumed non-empty at this point;
            caller already filters empty ones).
          language_model: the LLaMA model (e.g. self._llava_model.language_model
            from VADLLaVA). Frozen base + LoRA(q_proj, v_proj) applied.
          device: target torch device for the returned vec (caller's compute device)
          dtype: target torch dtype (e.g. fp16) for the returned vec

        Returns:
          text_vec  [B, 4096]  in `dtype`
        """
        tok = self._llava_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors='pt',
        )
        # Resolve language_model's actual device dynamically.
        # PEFT-wrapped models sometimes have inner weights on CPU even after
        # parent `.cuda()` (e.g. base_model.model.embed_tokens may not have been
        # touched). Find a leaf module whose device is reliable.
        # Try to get embed_tokens device specifically since that's what fires
        # on input_ids forward.
        try:
            embed = language_model.get_input_embeddings()
            lm_device = embed.weight.device
        except Exception:
            lm_device = next(language_model.parameters()).device
        # If language_model still has params on cpu, force-move to the caller's
        # compute device (where ego_feats live).
        if lm_device.type == 'cpu' and device.type == 'cuda':
            language_model.to(device)
            lm_device = device
        input_ids = tok['input_ids'].to(lm_device)
        attention_mask = tok['attention_mask'].to(lm_device)

        out = language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]      # [B, L, H]  (H = 4096 for LLaMA-7b)

        # masked mean-pool
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)   # [B, L, 1]
        text_vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # CRITICAL: move back to caller's compute device (not lm_device).
        # When PEFT layers ended up on CPU but our planner submodules are on
        # cuda, returning on lm_device causes downstream addmm device mismatch.
        return text_vec.to(device=device, dtype=dtype)

    # ----------------------------- forward -----------------------------

    def forward(
        self,
        ego_instructions: List[str],
        ego_feats: torch.Tensor,
        language_model,
        **kwargs,    # absorb optional FiLM-only kwargs (agent_query, agent_mask)
    ) -> torch.Tensor:
        """Compute α · delta with shape matching ego_fut_preds.

        Args:
          ego_instructions: list of B strings (may contain empty "" → skipped)
          ego_feats: [B, 1, ego_feat_dim]  or  [B, ego_feat_dim]
          language_model: the LLaMA causal-LM module (frozen base + LoRA)

        Returns:
          alpha_times_delta  [B, ego_fut_mode, fut_ts, 2]
        """
        if ego_feats.dim() == 3:
            ego_flat = ego_feats.squeeze(1)
        else:
            ego_flat = ego_feats
        B = ego_flat.shape[0]
        device = ego_flat.device
        dtype = ego_flat.dtype

        # Lazy build MLP once we know ego_feat_dim.
        if self.mlp is None:
            self._build_mlp(int(ego_flat.shape[-1]))
            self._move_to(device, dtype=self.text_proj.weight.dtype)

        # Device guard: ensure all our trainable submodules live on the same
        # device as ego_feats. Parent model.cuda() sometimes fails to reach
        # PEFT-wrapped + nested ModuleDicts; this is a cheap one-time move.
        if self.text_proj.weight.device != device:
            self.text_proj.to(device)
            if self.mlp is not None:
                self.mlp.to(device)
            with torch.no_grad():
                self.alpha.data = self.alpha.data.to(device)

        # Bail out cheaply if all instructions empty (e.g. --no-language pass).
        non_empty_flags = [bool(s and isinstance(s, str) and s.strip())
                           for s in ego_instructions]
        if not any(non_empty_flags):
            return torch.zeros(
                B, self.ego_fut_mode, self.fut_ts, 2,
                device=device, dtype=dtype,
            )
        non_empty_mask = torch.tensor(
            non_empty_flags, device=device, dtype=torch.bool)

        # Only encode non-empty texts; reconstruct B-batched output afterward.
        keep_idx = non_empty_mask.nonzero(as_tuple=True)[0]
        texts_kept = [ego_instructions[i] for i in keep_idx.tolist()]
        text_vec_kept = self.encode_text(
            texts_kept, language_model,
            device=device, dtype=self.text_proj.weight.dtype,
        )    # [K, 4096]

        # Project & fuse with ego_feats (only kept samples)
        text_proj_kept = self.text_proj(text_vec_kept)         # [K, text_proj_dim]
        ego_kept = ego_flat[keep_idx].to(text_proj_kept.dtype)  # [K, ego_feat_dim]
        fused = torch.cat([text_proj_kept, ego_kept], dim=-1)  # [K, P+E]
        delta_flat_kept = self.mlp(fused)                       # [K, M*T*2]
        delta_kept = delta_flat_kept.reshape(
            -1, self.ego_fut_mode, self.fut_ts, 2)             # [K, M, T, 2]

        # Scatter back to full-batch shape with zeros for empty samples.
        delta_full = torch.zeros(
            B, self.ego_fut_mode, self.fut_ts, 2,
            device=device, dtype=delta_kept.dtype,
        )
        delta_full[keep_idx] = delta_kept

        # Apply gated scaling.
        return (self.alpha.to(delta_full.dtype) * delta_full).to(dtype)
