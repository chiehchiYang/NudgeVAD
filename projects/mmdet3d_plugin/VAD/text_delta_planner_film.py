"""TextDeltaPlannerFiLM — FiLM-modulated variant of TextDeltaPlanner.

vs TextDeltaPlanner (concat-then-MLP):
    text_vec [B,4096] → text_proj → text_proj[B, P]
    concat(text_proj, ego_feats) → MLP → delta

FiLM (per-channel multiplicative + additive modulation):
    text_vec [B,4096] → text_proj → text_proj[B, P]
    text_proj → γ_proj → γ[B, ego_feat_dim]   # init weight=0, bias=1 → γ ≡ 1
    text_proj → β_proj → β[B, ego_feat_dim]   # init weight=0, bias=0 → β ≡ 0
    modulated_ego = γ * ego_feats + β
    modulated_ego → MLP → delta[B, M, T, 2]
    output = ego_fut_preds + alpha * delta     # alpha_init=0

Starting-point safety (same philosophy as α-gate):
  γ ≡ 1 and β ≡ 0 and α = 0 at init → first iter output = baseline.
  Each piece has its own "off-switch" init so any single failure mode is benign.

Ablation variants are produced by flipping these flags in the config:
  use_beta=False   → γ-only (additive removed)
  use_alpha=False  → γ-gate already enough (γ_init=1 keeps no-op start)
  use_concat=True  → hybrid: concat(text_proj, modulated_ego) before MLP
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class TextDeltaPlannerFiLM(nn.Module):
    def __init__(
        self,
        llava_tokenizer,
        ego_feat_dim: int,
        text_hidden_size: int = 4096,
        ego_fut_mode: int = 3,
        fut_ts: int = 12,
        text_proj_dim: int = 256,
        mlp_hidden_dim: int = 512,
        text_max_length: int = 64,
        alpha_init: float = 0.0,
        mlp_last_init_std: float = 0.01,
        # FiLM-specific knobs (ablation switches)
        use_gamma: bool = True,
        use_beta: bool = True,
        use_alpha: bool = True,
        use_concat: bool = False,
        # Level-2 FDE recipe: auxiliary endpoint head supervised by GT t=6s.
        use_endpoint_head: bool = False,
        endpoint_pull_init: float = 1.0,
        endpoint_hidden_dim: int = 256,
        # Goal-conditioned FiLM: predicted goal_xy projected to text_proj_dim
        # and ADDED to text_proj before producing γ/β. This makes γ/β depend
        # on both raw text and predicted goal → "double-conditioned" modulation.
        # Requires use_endpoint_head=True.
        use_goal_conditioning: bool = False,
        # User-spec "Explicit Goal Prediction" variant:
        goal_head_text_only: bool = False,
        goal_concat_to_mlp: bool = False,
        # Agent-aware cross-attention: text → query, agent_query → K/V.
        # Targets dynamic_only instructions ("follow the truck") that current
        # FiLM/concat can't ground onto the right agent. Output agent_attn_feat
        # is concat'd to MLP input.
        use_agent_attn: bool = False,
        agent_attn_dim: int = 256,       # internal attention hidden dim
        agent_query_dim: int = None,     # input agent feature dim from VADHead (auto-detected)
        # Speed-class auxiliary head:text_proj → K-bin softmax for ego avg speed
        # Auxiliary cross-entropy supervision only (no direct trajectory effect)
        # Forces text encoding to be speed-aware → cleaner signal for
        # "stop / slow / go fast" instructions → improved FDE.
        use_speed_class_head: bool = False,
        speed_num_classes: int = 7,
    ):
        super().__init__()
        self._llava_tokenizer = llava_tokenizer

        self.text_hidden_size = int(text_hidden_size)
        self.ego_fut_mode = int(ego_fut_mode)
        self.fut_ts = int(fut_ts)
        self.ego_feat_dim = int(ego_feat_dim)
        self.text_proj_dim = int(text_proj_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.text_max_length = int(text_max_length)
        self.use_gamma = bool(use_gamma)
        self.use_beta = bool(use_beta)
        self.use_alpha = bool(use_alpha)
        self.use_concat = bool(use_concat)
        self.use_endpoint_head = bool(use_endpoint_head)
        self.endpoint_hidden_dim = int(endpoint_hidden_dim)
        self.use_goal_conditioning = bool(use_goal_conditioning) and self.use_endpoint_head
        self.goal_head_text_only = bool(goal_head_text_only) and self.use_endpoint_head
        self.goal_concat_to_mlp = bool(goal_concat_to_mlp) and self.use_endpoint_head
        self.use_agent_attn = bool(use_agent_attn)
        self.agent_attn_dim = int(agent_attn_dim)
        # agent_query_dim is lazy-built if None (read from first forward)
        self._agent_query_dim = agent_query_dim
        self.use_speed_class_head = bool(use_speed_class_head)
        self.speed_num_classes = int(speed_num_classes)

        # Text projection (frozen-then-trainable)
        self.text_proj = nn.Linear(self.text_hidden_size, self.text_proj_dim)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)

        # FiLM γ: weight 0, bias 1 → γ ≡ 1 at init (no-op modulation)
        if self.use_gamma:
            self.gamma_proj = nn.Linear(self.text_proj_dim, self.ego_feat_dim)
            nn.init.zeros_(self.gamma_proj.weight)
            nn.init.ones_(self.gamma_proj.bias)
        else:
            self.gamma_proj = None

        # FiLM β: weight 0, bias 0 → β ≡ 0 at init (no shift)
        if self.use_beta:
            self.beta_proj = nn.Linear(self.text_proj_dim, self.ego_feat_dim)
            nn.init.zeros_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)
        else:
            self.beta_proj = None

        # MLP input dim depends on whether we also concat text_proj alongside
        # and whether we concat predicted goal_xy per mode
        mlp_in_dim = self.ego_feat_dim + (self.text_proj_dim if self.use_concat else 0)
        if self.goal_concat_to_mlp:
            mlp_in_dim += self.ego_fut_mode * 2   # goal_xy per mode, flattened
        if self.use_agent_attn:
            mlp_in_dim += self.agent_attn_dim     # agent-attn pooled feature
        out_dim = self.ego_fut_mode * self.fut_ts * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, out_dim),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        # MLP last layer: when α-gate is present, small random init (std=0.01) is
        # safe because α=0 zeros it out. When α-gate is removed (use_alpha=False),
        # the only thing keeping first-iter output = baseline is this last-layer
        # init — random std=0.01 leaks ~0.38 m bias. Force exactly zero in that
        # case so v4 also starts bit-identical to baseline.
        if self.use_alpha:
            nn.init.normal_(self.mlp[-1].weight, std=mlp_last_init_std)
        else:
            nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # α gate. When use_alpha=False we still register a buffer-like Parameter
        # at value 1 so the rest of the math is unchanged; γ_init=1 + β_init=0
        # are sufficient to keep first-iter output = baseline even without α.
        init_val = float(alpha_init) if self.use_alpha else 1.0
        self.alpha = nn.Parameter(torch.tensor([init_val], dtype=torch.float32),
                                  requires_grad=self.use_alpha)

        # Endpoint head: language directly predicts the 6s goal position.
        # Input: concat(text_proj, ego_feats) — same as concat-MLP path.
        # Output: [ego_fut_mode * 2] = (x,y) endpoint per cmd-mode.
        # Init: last-layer zero so output = 0 at start → no effect on baseline.
        # endpoint_pull: trainable gate; init 1.0 because the head's own zero
        # output already guarantees safe start (no need for second gate at 0).
        if self.use_endpoint_head:
            # Input: text_proj only (text-only goal head) or text_proj + ego_feats
            ep_in_dim = self.text_proj_dim if self.goal_head_text_only \
                else (self.text_proj_dim + self.ego_feat_dim)
            self.endpoint_head = nn.Sequential(
                nn.Linear(ep_in_dim, self.endpoint_hidden_dim),
                nn.LayerNorm(self.endpoint_hidden_dim),
                nn.GELU(),
                nn.Linear(self.endpoint_hidden_dim, self.ego_fut_mode * 2),
            )
            nn.init.normal_(self.endpoint_head[0].weight, std=0.02)
            nn.init.zeros_(self.endpoint_head[0].bias)
            nn.init.zeros_(self.endpoint_head[-1].weight)
            nn.init.zeros_(self.endpoint_head[-1].bias)
            self.endpoint_pull = nn.Parameter(
                torch.tensor([float(endpoint_pull_init)], dtype=torch.float32))
        else:
            self.endpoint_head = None
            self.endpoint_pull = None

        # Goal-conditioning projection: goal_xy [B, 2] → text_proj_dim feature
        # added to text_proj before γ/β projection.
        # Init: weight=0, bias=0 → goal_emb=0 → cond=text_proj → identical to
        # no-goal-conditioning at start (baseline preserved).
        if self.use_goal_conditioning:
            self.goal_proj = nn.Linear(2, self.text_proj_dim)
            nn.init.zeros_(self.goal_proj.weight)
            nn.init.zeros_(self.goal_proj.bias)
        else:
            self.goal_proj = None

        # Agent-aware cross-attention: text → agent K/V → attn → concat to MLP
        # Lazy-build agent_query_proj on first forward when we know agent_query
        # feature dim from VADHead.
        if self.use_agent_attn:
            # Text → query projection. Init: weight=0 → query=0 → attn=uniform
            # → softmax over K → output ≈ mean(V). With agent_out_proj zero-init
            # below, the final attn feat is 0 → MLP input goal columns are 0 →
            # delta unchanged at init.
            self.agent_q_proj = nn.Linear(self.text_proj_dim, self.agent_attn_dim)
            nn.init.zeros_(self.agent_q_proj.weight)
            nn.init.zeros_(self.agent_q_proj.bias)
            # Agent K and V projections (lazy build, see _build_agent_attn).
            self.agent_kv_built = False
            self.agent_k_proj = None
            self.agent_v_proj = None
            # Output projection. Zero-init to guarantee delta=0 at start.
            self.agent_out_proj = nn.Linear(self.agent_attn_dim, self.agent_attn_dim)
            nn.init.zeros_(self.agent_out_proj.weight)
            nn.init.zeros_(self.agent_out_proj.bias)
        else:
            self.agent_q_proj = None
            self.agent_k_proj = None
            self.agent_v_proj = None
            self.agent_out_proj = None
            self.agent_kv_built = True

        # Speed-class head: text_proj → K-way logits
        # Init: weight std=0.02 (regular) so we don't fight CE loss with zero init;
        # CE loss can absorb random init from epoch 1.
        if self.use_speed_class_head:
            self.speed_class_head = nn.Linear(self.text_proj_dim,
                                              self.speed_num_classes)
            nn.init.normal_(self.speed_class_head.weight, std=0.02)
            nn.init.zeros_(self.speed_class_head.bias)
        else:
            self.speed_class_head = None

    # ----------------------------- text encode -----------------------------

    def encode_text(self, texts: List[str], language_model, device,
                    dtype) -> torch.Tensor:
        tok = self._llava_tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.text_max_length, return_tensors='pt',
        )
        try:
            embed = language_model.get_input_embeddings()
            lm_device = embed.weight.device
        except Exception:
            lm_device = next(language_model.parameters()).device
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
        hidden = out.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        text_vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return text_vec.to(device=device, dtype=dtype)

    # ----------------------------- forward -----------------------------

    def _build_agent_kv(self, agent_query_dim: int, device, dtype):
        """Lazy-build agent K/V projections once we know the agent feature dim."""
        self.agent_k_proj = nn.Linear(agent_query_dim, self.agent_attn_dim).to(device=device, dtype=dtype)
        self.agent_v_proj = nn.Linear(agent_query_dim, self.agent_attn_dim).to(device=device, dtype=dtype)
        # Standard init for K (random) and V (zero) — zero V → attn output = 0 at init.
        nn.init.normal_(self.agent_k_proj.weight, std=0.02)
        nn.init.zeros_(self.agent_k_proj.bias)
        nn.init.zeros_(self.agent_v_proj.weight)
        nn.init.zeros_(self.agent_v_proj.bias)
        self.agent_kv_built = True
        self._agent_query_dim = agent_query_dim

    def forward(
        self,
        ego_instructions: List[str],
        ego_feats: torch.Tensor,
        language_model,
        agent_query: Optional[torch.Tensor] = None,    # [B, num_agents, D]
        agent_mask: Optional[torch.Tensor] = None,     # [B, num_agents] True=padded
    ) -> torch.Tensor:
        """Returns delta [B, M, T, 2].

        Side effect: when ``use_endpoint_head=True``, the per-sample predicted
        endpoint is stored on ``self.last_endpoint`` (shape [B, M, 2], or None
        for samples whose instruction was empty). The caller can read this via
        ``planner.last_endpoint`` after forward() returns. We keep the public
        forward return type as Tensor for backward compatibility with the
        existing ``_apply_text_delta`` call site.
        """
        self.last_endpoint = None  # reset each forward
        if ego_feats.dim() == 3:
            ego_flat = ego_feats.squeeze(1)
        else:
            ego_flat = ego_feats
        B = ego_flat.shape[0]
        device = ego_flat.device
        dtype = ego_flat.dtype

        # Device guard (same as plain TextDeltaPlanner)
        if self.text_proj.weight.device != device:
            self.text_proj.to(device)
            if self.gamma_proj is not None:
                self.gamma_proj.to(device)
            if self.beta_proj is not None:
                self.beta_proj.to(device)
            self.mlp.to(device)
            with torch.no_grad():
                self.alpha.data = self.alpha.data.to(device)
            if self.endpoint_head is not None:
                self.endpoint_head.to(device)
                with torch.no_grad():
                    self.endpoint_pull.data = self.endpoint_pull.data.to(device)
            if self.goal_proj is not None:
                self.goal_proj.to(device)
            # Agent-attn submodules also need device alignment
            if self.agent_q_proj is not None:
                self.agent_q_proj.to(device)
            if self.agent_out_proj is not None:
                self.agent_out_proj.to(device)
            if self.agent_k_proj is not None:
                self.agent_k_proj.to(device)
            if self.agent_v_proj is not None:
                self.agent_v_proj.to(device)

        non_empty_flags = [bool(s and isinstance(s, str) and s.strip())
                           for s in ego_instructions]
        if not any(non_empty_flags):
            return torch.zeros(
                B, self.ego_fut_mode, self.fut_ts, 2,
                device=device, dtype=dtype,
            )
        non_empty_mask = torch.tensor(
            non_empty_flags, device=device, dtype=torch.bool)
        keep_idx = non_empty_mask.nonzero(as_tuple=True)[0]
        texts_kept = [ego_instructions[i] for i in keep_idx.tolist()]

        text_vec_kept = self.encode_text(
            texts_kept, language_model,
            device=device, dtype=self.text_proj.weight.dtype,
        )    # [K, 4096]
        text_proj_kept = self.text_proj(text_vec_kept)             # [K, P]
        ego_kept = ego_flat[keep_idx].to(text_proj_kept.dtype)     # [K, E]

        # ---- Speed class auxiliary head ----
        # Avoid in-place index assign (`full[keep_idx] = ...`) which interacts
        # badly with autograd on subsequent iters. Just store the K-row logits
        # and the keep_idx separately — VAD_llava loss code masks GT to match.
        self.last_speed_logits = None
        self.last_speed_keep_idx = None
        if self.speed_class_head is not None:
            self.speed_class_head.to(device)
            speed_input = text_proj_kept.detach()
            logits_kept = self.speed_class_head(speed_input)   # [K, num_classes]
            self.last_speed_logits = logits_kept
            self.last_speed_keep_idx = keep_idx

        # ---- Step 1: predict goal_xy first (needed for goal-conditioned FiLM) ----
        endpoint_kept = None
        if self.endpoint_head is not None:
            # Text-only goal head: input is text_proj alone (pure language→goal)
            if self.goal_head_text_only:
                ep_input = text_proj_kept
            else:
                ep_input = torch.cat([text_proj_kept, ego_kept], dim=-1)
            endpoint_kept = self.endpoint_head(ep_input).reshape(
                -1, self.ego_fut_mode, 2)             # [K, M, 2]

        # ---- Step 2: build conditioning vector (text + optional goal) ----
        cond_kept = text_proj_kept                                  # [K, P]
        if self.use_goal_conditioning and endpoint_kept is not None:
            # Pool goal over modes (mean) → [K, 2] → goal_emb [K, P]
            goal_mean = endpoint_kept.mean(dim=1)                   # [K, 2]
            goal_emb = self.goal_proj(goal_mean)                    # [K, P]
            cond_kept = cond_kept + goal_emb                        # double condition

        # ---- Step 3: FiLM modulation using (cond) ----
        modulated = ego_kept
        if self.gamma_proj is not None:
            gamma = self.gamma_proj(cond_kept)                      # [K, E]
            modulated = gamma * modulated
        if self.beta_proj is not None:
            beta = self.beta_proj(cond_kept)                        # [K, E]
            modulated = modulated + beta

        if self.use_concat:
            fused = torch.cat([cond_kept, modulated], dim=-1)
        else:
            fused = modulated
        # Concat predicted goal_xy (flattened per mode) so MLP sees the goal
        # explicitly at every step (user's "Explicit Goal Prediction" recipe).
        if self.goal_concat_to_mlp and endpoint_kept is not None:
            goal_flat = endpoint_kept.reshape(-1, self.ego_fut_mode * 2)
            fused = torch.cat([fused, goal_flat], dim=-1)

        # Agent-aware cross-attention: text → query, agent features → K/V.
        # When agent_attn is enabled, ALWAYS append the agent_attn_dim columns
        # to MLP input — even when agent_query is None (eval path edge case) —
        # so MLP's first-layer input dim is consistent between train and eval.
        if self.use_agent_attn:
            if agent_query is not None:
                # Move agent_query/mask to compute device — they may be on CPU
                # (VAD's simple_test_pts does .cpu() on outs before bbox_result).
                agent_query = agent_query.to(device=device)
                if not self.agent_kv_built:
                    self._build_agent_kv(int(agent_query.shape[-1]),
                                         device=device,
                                         dtype=self.text_proj.weight.dtype)
                agent_kept_q = agent_query[keep_idx].to(
                    device=device, dtype=self.text_proj.weight.dtype)
                q = self.agent_q_proj(cond_kept).unsqueeze(1)
                k = self.agent_k_proj(agent_kept_q)
                v = self.agent_v_proj(agent_kept_q)
                scale = float(self.agent_attn_dim) ** -0.5
                scores = torch.matmul(q, k.transpose(-1, -2)) * scale
                if agent_mask is not None:
                    agent_mask = agent_mask.to(device=device)
                    mk = agent_mask[keep_idx]
                    scores = scores.masked_fill(mk.unsqueeze(1), float('-inf'))
                attn = torch.softmax(scores, dim=-1)
                attn_out = torch.matmul(attn, v).squeeze(1)
                attn_out = self.agent_out_proj(attn_out)
                self.last_agent_attn = attn
            else:
                # No agent_query supplied — pad with zeros so MLP input dim
                # matches train-time shape. (Common in standalone unit tests
                # or eval paths that didn't thread agent_query through.)
                attn_out = torch.zeros(
                    cond_kept.shape[0], self.agent_attn_dim,
                    device=device, dtype=cond_kept.dtype,
                )
                self.last_agent_attn = None
            fused = torch.cat([fused, attn_out], dim=-1)

        delta_flat_kept = self.mlp(fused)
        delta_kept = delta_flat_kept.reshape(
            -1, self.ego_fut_mode, self.fut_ts, 2)

        delta_full = torch.zeros(
            B, self.ego_fut_mode, self.fut_ts, 2,
            device=device, dtype=delta_kept.dtype,
        )
        delta_full[keep_idx] = delta_kept

        # ---- Step 4: endpoint additive residual on final step ----
        if endpoint_kept is not None:
            endpoint_full = torch.zeros(
                B, self.ego_fut_mode, 2,
                device=device, dtype=endpoint_kept.dtype,
            )
            endpoint_full[keep_idx] = endpoint_kept
            pull = self.endpoint_pull.to(endpoint_full.dtype)
            delta_full[..., -1, :] = delta_full[..., -1, :] + pull * endpoint_full
            self.last_endpoint = endpoint_full
        else:
            self.last_endpoint = None

        alpha = self.alpha.to(delta_full.dtype)
        return (alpha * delta_full).to(dtype)
