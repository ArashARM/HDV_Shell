import os
import math 
import random
import torch
import torch.nn as nn

class PPNet(nn.Module):
    """
    PPNet: refines given initial seed UVs with learned offsets (stable init),
    and predicts global params (w/h) + ring2 coeffs from context.
    Returns keys compatible with your current training loop:
      pred["seeds_raw"]         : (B,S,2)
      pred["w_raw"]             : (B,)
      pred["h_raw"]             : (B,)
      pred["ring2_alpha_raw"]   : (B,K)   if ring2_K>0
      pred["gate_probs"]        : (B,S)   optional gating
    """
    def __init__(self, context_dim, n_seeds, use_anisotropy=False, hidden=256):
        super().__init__()
        self.n_seeds = n_seeds
        self.use_anisotropy = use_anisotropy

        # trunk
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # per-seed refinement conditioned on context + uv_init
        self.seed_refine = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # offset per seed
        self.delta_head = nn.Linear(hidden, 2)

        # optional gating for "how many seeds used"
        self.gate_head = nn.Linear(hidden, 1)

        # global params
        self.w_head = nn.Linear(hidden, 1)
        self.h_head = nn.Linear(hidden, 1)

        if use_anisotropy:
            self.theta_head = nn.Linear(hidden, 1)
            self.a_head = nn.Linear(hidden, 1)

    def forward(self, context, uv_init, offset_scale=1, clamp01=True):
        """
        context : (B,C)
        uv_init : (S,2) or (B,S,2)

        New behavior:
        - refinement is done in a smooth bounded way
        - instead of hard clamping in UV space, we move in logit space:
              seeds = sigmoid(logit(uv_init) + offset_scale * tanh(delta_raw))
        - this preserves bounded seeds in [0,1] while keeping gradients smooth
        """
        B = context.shape[0]
        S = self.n_seeds
        eps = 1e-6

        # ensure uv_init is (B,S,2)
        if uv_init.dim() == 2:
            uv_init_b = uv_init.unsqueeze(0).expand(B, -1, -1)
        elif uv_init.dim() == 3:
            uv_init_b = uv_init
        else:
            raise ValueError("uv_init must have shape (S,2) or (B,S,2)")

        if uv_init_b.shape[1:] != (S, 2):
            raise ValueError(f"Expected uv_init (B,{S},2) or ({S},2), got {tuple(uv_init_b.shape)}")

        z = self.mlp(context)                          # (B,H)
        z_rep = z.unsqueeze(1).expand(-1, S, -1)      # (B,S,H)

        seed_in = torch.cat([z_rep, uv_init_b], dim=-1)   # (B,S,H+2)
        h = self.seed_refine(seed_in)                     # (B,S,H)

        delta_raw = self.delta_head(h)                    # (B,S,2)
        delta = offset_scale * torch.tanh(delta_raw)      # bounded smooth latent offset

        # Smooth bounded parameterization:
        # map uv_init to logit space, add offset, map back with sigmoid
        uv_safe = uv_init_b.clamp(eps, 1.0 - eps)
        uv_logits = torch.log(uv_safe) - torch.log(1.0 - uv_safe)
        seeds_raw = torch.sigmoid(uv_logits + delta)

        # keep backward compatibility with older call sites;
        # no hard clamp is needed anymore
        if clamp01:
            seeds_raw = seeds_raw

        gate_logits = self.gate_head(h).squeeze(-1)       # (B,S)
        gate_probs = torch.sigmoid(gate_logits)           # (B,S)

        w_raw = self.w_head(z).view(-1)                   # (B,)
        h_raw = self.h_head(z).view(-1) if hasattr(self, "h_head") else None

        out = {
            "uv_init": uv_init_b,
            "seeds_raw": seeds_raw,
            "delta_raw": delta_raw,
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
            "w_raw": w_raw,
            "h_raw": h_raw,
        }

        if self.use_anisotropy:
            out["theta"] = self.theta_head(h).squeeze(-1)   # (B,S)
            out["a_raw"] = self.a_head(h).squeeze(-1)       # (B,S)

        return out