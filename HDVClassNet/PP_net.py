import torch
import torch.nn as nn


class PPNet(nn.Module):
    """
    PPNet: predicts all learnable decoder controls.

    Returns possible keys:
      pred["seeds_raw"]            : (B,S,2)
      pred["w_raw"]                : (B,S,S)
      pred["h_raw"]                : (B,)         if predict_height=True
      pred["theta"]                : (B,S)        if anisotropy enabled
      pred["a_raw"]                : (B,S)        if anisotropy enabled
      pred["gate_logits"]          : (B,S)        if use_gating=True
      pred["gate_probs"]           : (B,S)        if use_gating=True

      pred["boundary_width_raw"]   : (B,)         if predict_boundary_params=True
      pred["boundary_alpha_raw"]   : (B,)         if predict_boundary_params=True
      pred["boundary_beta_raw"]    : (B,)         if predict_boundary_params=True
    """

    def __init__(
        self,
        context_dim,
        n_seeds,
        use_Metric_anisotropy=False,
        predict_height=False,
        use_gating=False,
        predict_boundary_width=False,
        hidden=256,
        freeze_w=False,
        w_const=0.25,
        eps_uv=1e-4,
        max_delta_logit=0.30,
        max_step_uv=0.08,
        gate_bias_init=0.0,
    ):
        super().__init__()
        self.n_seeds = n_seeds
        self.use_Metric_anisotropy = use_Metric_anisotropy
        self.predict_height = predict_height
        self.use_gating = use_gating
        self.predict_boundary_width = predict_boundary_width

        self.freeze_w = freeze_w
        self.w_const = w_const

        self.eps_uv = eps_uv
        self.max_delta_logit = max_delta_logit
        self.max_step_uv = max_step_uv
        self.gate_bias_init = gate_bias_init

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

        # global heads
        self.w_head = nn.Linear(hidden, 1)

        if self.predict_height:
            self.h_head = nn.Linear(hidden, 1)

        if self.predict_boundary_width:
            self.boundary_width_head = nn.Linear(hidden, 1)

        if self.use_gating:
            self.gate_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, self.gate_bias_init)

        if self.use_Metric_anisotropy:
            self.theta_head = nn.Linear(hidden, 1)
            self.a_head = nn.Linear(hidden, 1)

    def forward(self, context, uv_init, offset_scale=1.0):
        B = context.shape[0]
        S = self.n_seeds

        eps_uv = self.eps_uv
        max_delta_logit = self.max_delta_logit
        max_step_uv = self.max_step_uv

        if uv_init.dim() == 2:
            uv_init_b = uv_init.unsqueeze(0).expand(B, -1, -1)
        elif uv_init.dim() == 3:
            uv_init_b = uv_init
        else:
            raise ValueError("uv_init must have shape (S,2) or (B,S,2)")

        if uv_init_b.shape[1:] != (S, 2):
            raise ValueError(
                f"Expected uv_init (B,{S},2) or ({S},2), got {tuple(uv_init_b.shape)}"
            )

        if not torch.isfinite(context).all():
            raise RuntimeError("PPNet input 'context' is non-finite")
        if not torch.isfinite(uv_init_b).all():
            raise RuntimeError("PPNet input 'uv_init' is non-finite")

        uv_base = uv_init_b.clamp(eps_uv, 1.0 - eps_uv)
        if not torch.isfinite(uv_base).all():
            raise RuntimeError("PPNet produced non-finite uv_base")

        z = self.mlp(context)
        if not torch.isfinite(z).all():
            raise RuntimeError("PPNet produced non-finite z from mlp(context)")

        z_rep = z.unsqueeze(1).expand(-1, S, -1)
        seed_in = torch.cat([z_rep, uv_base], dim=-1)
        if not torch.isfinite(seed_in).all():
            raise RuntimeError("PPNet produced non-finite seed_in")

        h = self.seed_refine(seed_in)
        if not torch.isfinite(h).all():
            raise RuntimeError("PPNet produced non-finite h from seed_refine")

        delta_raw = self.delta_head(h)
        if not torch.isfinite(delta_raw).all():
            raise RuntimeError("PPNet produced non-finite delta_raw")

        delta_logit = max_delta_logit * offset_scale * torch.tanh(delta_raw)
        if not torch.isfinite(delta_logit).all():
            raise RuntimeError("PPNet produced non-finite delta_logit")

        uv_logits = torch.logit(uv_base, eps=eps_uv)
        if not torch.isfinite(uv_logits).all():
            raise RuntimeError("PPNet produced non-finite uv_logits")

        seeds_uv = torch.sigmoid(uv_logits + delta_logit)
        if not torch.isfinite(seeds_uv).all():
            raise RuntimeError("PPNet produced non-finite seeds_uv before trust region")

        delta_uv = seeds_uv - uv_base
        if not torch.isfinite(delta_uv).all():
            raise RuntimeError("PPNet produced non-finite delta_uv before clamp")

        delta_uv = delta_uv.clamp(-max_step_uv, max_step_uv)
        seeds_uv = (uv_base + delta_uv).clamp(eps_uv, 1.0 - eps_uv)

        if not torch.isfinite(seeds_uv).all():
            raise RuntimeError("PPNet produced non-finite seeds_uv after trust region")

        out = {
            "uv_init": uv_base,
            "seeds_raw": seeds_uv,
            "delta_raw": delta_raw,
            "delta_logit": delta_logit,
            "delta_uv": delta_uv,
        }

        if self.freeze_w:
            out["w_raw"] = torch.full(
                (B, S, S),
                self.w_const,
                device=z.device,
                dtype=z.dtype,
            )
        else:
            pair_h = 0.5 * (h.unsqueeze(2) + h.unsqueeze(1))
            w_raw = self.w_head(pair_h).squeeze(-1)
            out["w_raw"] = 0.5 * (w_raw + w_raw.transpose(1, 2))

        if not torch.isfinite(out["w_raw"]).all():
            raise RuntimeError("PPNet produced non-finite w_raw")

        if self.predict_height:
            out["h_raw"] = self.h_head(z).view(-1)
            if not torch.isfinite(out["h_raw"]).all():
                raise RuntimeError("PPNet produced non-finite h_raw")

        if self.predict_boundary_width:
            out["boundary_width_raw"] = self.boundary_width_head(z).view(-1)
            if not torch.isfinite(out["boundary_width_raw"]).all():
                raise RuntimeError("PPNet produced non-finite boundary_width_raw")

        if self.use_gating:
            gate_logits = self.gate_head(h).squeeze(-1)
            if not torch.isfinite(gate_logits).all():
                raise RuntimeError("PPNet produced non-finite gate_logits")
            out["gate_logits"] = gate_logits
            out["gate_probs"] = torch.sigmoid(gate_logits)

        if self.use_Metric_anisotropy:
            theta = self.theta_head(h).squeeze(-1)
            a_raw = self.a_head(h).squeeze(-1)
            if not torch.isfinite(theta).all():
                raise RuntimeError("PPNet produced non-finite theta")
            if not torch.isfinite(a_raw).all():
                raise RuntimeError("PPNet produced non-finite a_raw")
            out["theta"] = theta
            out["a_raw"] = a_raw

        return out
