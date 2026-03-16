import torch
import torch.nn as nn


class PPNet(nn.Module):
    """
    PPNet: refines initial seed UVs and predicts global geometric width.

    Returns:
      pred["seeds_raw"] : (B,S,2)
      pred["w_raw"]     : (B,)
      pred["theta"]     : (B,S)   if anisotropy enabled
      pred["a_raw"]     : (B,S)   if anisotropy enabled
    """

    def __init__(
        self,
        context_dim,
        n_seeds,
        use_anisotropy=False,
        predict_height=False,
        use_gating=False,
        hidden=256,
    ):
        super().__init__()
        self.n_seeds = n_seeds
        self.use_anisotropy = use_anisotropy
        self.predict_height = predict_height
        self.use_gating = use_gating

        # global context trunk
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # per-seed refinement
        self.seed_refine = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.delta_head = nn.Linear(hidden, 2)

        # geometric width latent
        self.w_head = nn.Linear(hidden, 1)

        if self.predict_height:
            self.h_head = nn.Linear(hidden, 1)

        if self.use_gating:
            self.gate_head = nn.Linear(hidden, 1)

        if use_anisotropy:
            self.theta_head = nn.Linear(hidden, 1)
            self.a_head = nn.Linear(hidden, 1)

    def forward(self, context, uv_init, offset_scale=1.0):
        B = context.shape[0]
        S = self.n_seeds
        eps = 1e-6

        if uv_init.dim() == 2:
            uv_init_b = uv_init.unsqueeze(0).expand(B, -1, -1)
        elif uv_init.dim() == 3:
            uv_init_b = uv_init
        else:
            raise ValueError("uv_init must have shape (S,2) or (B,S,2)")

        if uv_init_b.shape[1:] != (S, 2):
            raise ValueError(f"Expected uv_init (B,{S},2) or ({S},2), got {tuple(uv_init_b.shape)}")

        z = self.mlp(context)                     # (B,H)
        z_rep = z.unsqueeze(1).expand(-1, S, -1) # (B,S,H)

        seed_in = torch.cat([z_rep, uv_init_b], dim=-1)
        h = self.seed_refine(seed_in)

        delta_raw = self.delta_head(h)
        delta = offset_scale * torch.tanh(delta_raw)

        # smooth bounded UV update in logit space
        uv_safe = uv_init_b.clamp(eps, 1.0 - eps)
        uv_logits = torch.log(uv_safe) - torch.log(1.0 - uv_safe)
        seeds_raw = torch.sigmoid(uv_logits + delta)

        out = {
            "uv_init": uv_init_b,
            "seeds_raw": seeds_raw,
            "delta_raw": delta_raw,
            "w_raw": self.w_head(z).view(-1),
        }

        if self.predict_height:
            out["h_raw"] = self.h_head(z).view(-1)

        if self.use_gating:
            gate_logits = self.gate_head(h).squeeze(-1)
            out["gate_logits"] = gate_logits
            out["gate_probs"] = torch.sigmoid(gate_logits)

        if self.use_anisotropy:
            out["theta"] = self.theta_head(h).squeeze(-1)
            out["a_raw"] = self.a_head(h).squeeze(-1)

        return out