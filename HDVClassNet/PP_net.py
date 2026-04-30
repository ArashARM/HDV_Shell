import math
import torch
import torch.nn as nn


class PPNet(nn.Module):
    """
    PPNet: predicts ALL VoronoiDecoder controls.

    Outputs always contain full control dictionary:
    - seeds_raw (required)
    - w_raw (required)
    - optional parameters returned as None if disabled
    """

    def __init__(
        self,
        context_dim,
        n_seeds,
        hidden=256,

        # feature toggles
        use_Metric_anisotropy=False,
        predict_height=False,
        predict_boundary_params=False,
        predict_tau=False,
        tau_pred_start=0.02,
        tau_pred_min=1e-4,
        tau_pred_max=0.2,

        # width behavior
        freeze_w=False,
        w_const=0.25,
        w_head_bias_init=0.0,

        # seed update constraints
        eps_uv=1e-4,
        max_delta_logit=0.30,
        max_step_uv=0.08,
        seed_id_dim=16,
        allow_seed_outside_domain=False,
        seed_domain_margin=0.25,

        # safety
        enable_checks=True,
    ):
        super().__init__()

        self.n_seeds = n_seeds
        self.use_Metric_anisotropy = use_Metric_anisotropy
        self.predict_height = predict_height
        self.predict_boundary_params = predict_boundary_params
        self.predict_tau = predict_tau
        self.tau_pred_start = float(tau_pred_start)
        self.tau_pred_min = float(tau_pred_min)
        self.tau_pred_max = float(tau_pred_max)

        self.freeze_w = freeze_w
        self.w_const = w_const

        self.eps_uv = eps_uv
        self.max_delta_logit = max_delta_logit
        self.max_step_uv = max_step_uv
        self.seed_id_dim = int(seed_id_dim)
        self.allow_seed_outside_domain = bool(allow_seed_outside_domain)
        self.seed_domain_margin = float(seed_domain_margin)

        self.enable_checks = enable_checks

        if self.predict_tau:
            if not (self.tau_pred_min > 0.0):
                raise ValueError(f"tau_pred_min must be > 0, got {self.tau_pred_min}")
            if not (self.tau_pred_max > self.tau_pred_min):
                raise ValueError(
                    f"tau_pred_max must be > tau_pred_min, got min={self.tau_pred_min}, max={self.tau_pred_max}"
                )
            if not (self.tau_pred_min <= self.tau_pred_start <= self.tau_pred_max):
                raise ValueError(
                    "tau_pred_start must lie within [tau_pred_min, tau_pred_max], "
                    f"got start={self.tau_pred_start}, min={self.tau_pred_min}, max={self.tau_pred_max}"
                )

        # -------------------------
        # trunk
        # -------------------------
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # -------------------------
        # per-seed refinement
        # -------------------------
        if self.seed_id_dim > 0:
            self.seed_id_embed = nn.Embedding(self.n_seeds, self.seed_id_dim)
        else:
            self.seed_id_embed = None

        self.seed_refine = nn.Sequential(
            nn.Linear(hidden + 2 + max(self.seed_id_dim, 0), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.delta_head = nn.Linear(hidden, 2)

        # -------------------------
        # heads
        # -------------------------
        self.w_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.w_head.weight)
        nn.init.constant_(self.w_head.bias, w_head_bias_init)

        if self.predict_height:
            self.h_head = nn.Linear(hidden, 1)

        if self.use_Metric_anisotropy:
            self.theta_head = nn.Linear(hidden, 1)
            self.a_head = nn.Linear(hidden, 1)

        if self.predict_boundary_params:
            self.boundary_width_head = nn.Linear(hidden, 1)
            self.boundary_alpha_head = nn.Linear(hidden, 1)
            self.boundary_beta_head = nn.Linear(hidden, 1)

        if self.predict_tau:
            self.tau_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.tau_head.weight)
            tau_range = self.tau_pred_max - self.tau_pred_min
            tau_frac = (self.tau_pred_start - self.tau_pred_min) / tau_range
            tau_frac = min(max(tau_frac, 1e-6), 1.0 - 1e-6)
            nn.init.constant_(self.tau_head.bias, math.log(tau_frac / (1.0 - tau_frac)))

    # -------------------------------------------------------
    # safety helper
    # -------------------------------------------------------

    def _check(self, tensor, name):
        if self.enable_checks and not torch.isfinite(tensor).all():
            raise RuntimeError(f"PPNet produced non-finite {name}")

    # -------------------------------------------------------
    # forward
    # -------------------------------------------------------

    def forward(self, context, uv_init, offset_scale=1.0):
        B = context.shape[0]
        S = self.n_seeds
        eps_uv = self.eps_uv

        self._check(context, "context")

        # -------------------------
        # prepare seeds
        # -------------------------
        if uv_init.dim() == 2:
            uv_init_b = uv_init.unsqueeze(0).expand(B, -1, -1)
        elif uv_init.dim() == 3:
            uv_init_b = uv_init
        else:
            raise ValueError("uv_init must be (S,2) or (B,S,2)")

        if self.allow_seed_outside_domain:
            uv_base = uv_init_b
        else:
            uv_base = uv_init_b.clamp(eps_uv, 1.0 - eps_uv)
        self._check(uv_base, "uv_base")

        # -------------------------
        # trunk
        # -------------------------
        z = self.mlp(context)
        self._check(z, "z")

        z_rep = z.unsqueeze(1).expand(-1, S, -1)
        if self.seed_id_embed is not None:
            seed_ids = torch.arange(S, device=uv_base.device, dtype=torch.long)
            seed_id_features = self.seed_id_embed(seed_ids).unsqueeze(0).expand(B, -1, -1)
            seed_in = torch.cat([z_rep, uv_base, seed_id_features], dim=-1)
        else:
            seed_in = torch.cat([z_rep, uv_base], dim=-1)
        self._check(seed_in, "seed_in")

        h = self.seed_refine(seed_in)
        self._check(h, "h")

        # -------------------------
        # seed refinement (bounded)
        # -------------------------
        delta_raw = self.delta_head(h)
        self._check(delta_raw, "delta_raw")

        # Apply residual updates directly in UV space so seeds near the domain
        # boundary can still move back inward without sigmoid/logit saturation.
        delta_dir = torch.tanh(delta_raw)
        self._check(delta_dir, "delta_dir")

        step_cap = torch.as_tensor(
            self.max_step_uv * offset_scale,
            device=uv_base.device,
            dtype=uv_base.dtype,
        )
        if self.allow_seed_outside_domain:
            delta_uv = delta_dir * step_cap
        else:
            room_lo = (uv_base - eps_uv).clamp_min(0.0)
            room_hi = (1.0 - eps_uv - uv_base).clamp_min(0.0)
            step_lo = torch.minimum(room_lo, step_cap)
            step_hi = torch.minimum(room_hi, step_cap)
            delta_uv = torch.where(
                delta_dir >= 0.0,
                delta_dir * step_hi,
                delta_dir * step_lo,
            )
        self._check(delta_uv, "delta_uv")

        seeds_uv = uv_base + delta_uv
        if self.allow_seed_outside_domain:
            margin = max(float(self.seed_domain_margin), 0.0)
            seeds_uv = seeds_uv.clamp(-margin, 1.0 + margin)
        else:
            seeds_uv = seeds_uv.clamp(eps_uv, 1.0 - eps_uv)
        self._check(seeds_uv, "seeds_uv_final")

        out = {
            "seeds_raw": seeds_uv,
        }

        # -------------------------
        # width
        # -------------------------
        if self.freeze_w:
            w_raw = torch.full((B, S, S), self.w_const, device=z.device, dtype=z.dtype)
        else:
            pair_h = 0.5 * (h.unsqueeze(2) + h.unsqueeze(1))
            w_raw = self.w_head(pair_h).squeeze(-1)
            w_raw = 0.5 * (w_raw + w_raw.transpose(1, 2))

        self._check(w_raw, "w_raw")
        out["w_raw"] = w_raw

        # -------------------------
        # height
        # -------------------------
        if self.predict_height:
            h_raw = self.h_head(z).view(-1)
            self._check(h_raw, "h_raw")
        else:
            h_raw = None
        out["h_raw"] = h_raw

        # -------------------------
        # anisotropy
        # -------------------------
        if self.use_Metric_anisotropy:
            theta = self.theta_head(h).squeeze(-1)
            a_raw = self.a_head(h).squeeze(-1)
            self._check(theta, "theta")
            self._check(a_raw, "a_raw")
        else:
            theta = None
            a_raw = None

        out["theta"] = theta
        out["a_raw"] = a_raw

        # -------------------------
        # boundary
        # -------------------------
        if self.predict_boundary_params:
            bw = self.boundary_width_head(z).view(-1)
            ba = self.boundary_alpha_head(z).view(-1)
            bb = self.boundary_beta_head(z).view(-1)

            self._check(bw, "boundary_width_raw")
            self._check(ba, "boundary_alpha_raw")
            self._check(bb, "boundary_beta_raw")
        else:
            bw = ba = bb = None

        out["boundary_width_raw"] = bw
        out["boundary_alpha_raw"] = ba
        out["boundary_beta_raw"] = bb

        # -------------------------
        # tau
        # -------------------------
        if self.predict_tau:
            tau_logits = self.tau_head(z).view(-1)
            tau = self.tau_pred_min + (self.tau_pred_max - self.tau_pred_min) * torch.sigmoid(tau_logits)
            self._check(tau, "tau")
        else:
            tau = None

        out["tau"] = tau

        return out
