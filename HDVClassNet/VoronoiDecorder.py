import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
class VoronoiDecoder(nn.Module):
    """
    Functional Voronoi "decoder" (no internal nn.Parameters for seeds/aniso/width/height/gates).
    You pass all trainable variables (predicted by ParamNet / optimizer) into forward().

    Key improvement:
      - Replace Q = 1 - sum(w_soft^2) with a *pair-gated* edge score that focuses on the
        "top-2 competition" idea *differentiably*:

            Q_pair(x) = sum_{i<j} [ g_gap(i,j,x) * g_big(i,j,x) * (w_i(x) w_j(x)) ]

        where:
          g_gap uses |w_i - w_j| (small gap => close competition => edge)
          g_big requires w_i and w_j to be sufficiently large (filters tiny weights)

    Inputs to forward():
      - points_uv: (N,2)
      - tau: soft Voronoi temperature (float)
      - seeds_raw: (S,2)  (you can change mapping in seeds_uv())
      - theta: (S,) (optional) anisotropy angle
      - a_raw: (S,) (optional) anisotropy log-parameter (tanh mapped to [a_min,a_max])
      - w_raw: scalar tensor controlling "thickness knob" (through Q_thr mapping)
      - h_raw: scalar tensor controlling height
      - points_face_id: (N,) long optional: face id per vertex/point

      New (optional) differentiable gating controls:
      - gap_thr_raw: scalar tensor -> maps to gap_thr in [gap_thr_min, gap_thr_max]
            (bigger gap_thr => more tolerant => thicker/stronger edges in Q_pair)
      - big_thr_raw: scalar tensor -> maps to big_thr in [big_thr_min, big_thr_max]
            (bigger big_thr => stricter "both weights must be big" => fewer edges)
      - alpha_raw: optional scalar controlling softness of gap gate (if you want learnable)
      - eta_raw: optional scalar controlling softness of big gate (if you want learnable)

    Returns:
      w_soft, d, M, seeds, rho, t_uv_raw, t_uv, h, Q_used
    """
    def __init__(
        self,
        n_seeds: int,
        eps: float = 1e-8,
        use_anisotropy: bool = True,
        # width controls (these control the *threshold* mapping, not physical mm directly)
        w_min: float = 0.01,
        w_max: float = 3,
        # adaptive Q_thr (smooth lo/hi)
        q_hi_temp: float = 0.02,
        q_lo_temp: float = 0.02,
        q_range_eps: float = 1e-4,
        beta: float = 0.02,
        raw_temp: float = 5.0,
        # height controls
        h_min: float = 0.50,
        h_max: float = 2.00,
        fixed_height: float | None = None,
        # boundary & periodicity
        boundary_solid_idx: torch.Tensor | None = None,
        face_u_periodic: torch.Tensor | None = None,
        face_v_periodic: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
        # --- new: pair-gated Q controls ---
        use_pair_gated_Q: bool = True,
        gap_thr_min: float = 0.00,
        gap_thr_max: float = 0.50,   # |w_i-w_j| is in [0,1]
        big_thr_min: float = 0.00,
        big_thr_max: float = 0.60,   # weights above ~0.2..0.4 are typically "big"
        alpha_min: float = 0.01,     # softness for gap gate
        alpha_max: float = 0.20,
        eta_min: float = 0.01,       # softness for big gate
        eta_max: float = 0.20,
        # fixed defaults if raw values not provided
        gap_thr_default: float = 0.15,
        big_thr_default: float = 0.10,
        alpha_default: float = 0.05,
        eta_default: float = 0.05,
        # --- adaptive quantile levels (smooth) ---
        p_lo: float = 0.70,      # interior-ish
        p_hi: float = 0.95,      # edge-ish
        q_smooth: float = 0.02,  # "CDF softness" in Q units (tune)
        q_newton_iters: int = 6,
        q_deriv_eps: float = 1e-6,
    ):
        super().__init__()
        self.n_seeds = int(n_seeds)
        self.eps = float(eps)
        self.use_anisotropy = bool(use_anisotropy)

        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.q_hi_temp = float(q_hi_temp)
        self.q_lo_temp = float(q_lo_temp)
        self.q_range_eps = float(q_range_eps)
        self.beta = float(beta)
        self.raw_temp = float(raw_temp)
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.q_smooth = float(q_smooth)
        self.q_newton_iters = int(q_newton_iters)
        self.q_deriv_eps = float(q_deriv_eps)

        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.fixed_height = fixed_height

        # pair-gated Q settings
        self.use_pair_gated_Q = bool(use_pair_gated_Q)
        self.gap_thr_min = float(gap_thr_min)
        self.gap_thr_max = float(gap_thr_max)
        self.big_thr_min = float(big_thr_min)
        self.big_thr_max = float(big_thr_max)

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)

        self.gap_thr_default = float(gap_thr_default)
        self.big_thr_default = float(big_thr_default)
        self.alpha_default = float(alpha_default)
        self.eta_default = float(eta_default)

        # buffers
        if boundary_solid_idx is None:
            boundary_solid_idx = torch.empty(0, dtype=torch.long)
        if face_u_periodic is None:
            face_u_periodic = torch.zeros(1, dtype=torch.bool)
        if face_v_periodic is None:
            face_v_periodic = torch.zeros(1, dtype=torch.bool)
        if seed_face_id is None:
            seed_face_id = torch.zeros(self.n_seeds, dtype=torch.long)

        self.register_buffer("boundary_solid_idx", boundary_solid_idx.to(torch.long))
        self.register_buffer("face_u_periodic", face_u_periodic.to(torch.bool))
        self.register_buffer("face_v_periodic", face_v_periodic.to(torch.bool))
        self.register_buffer("seed_face_id", seed_face_id.to(torch.long))

    # -------------------- parameter maps --------------------
    def seeds_uv(self, seeds_raw: torch.Tensor) -> torch.Tensor:
        # Keep as-is (your upstream can clamp/sigmoid if desired)
        return seeds_raw

    def width(self, w_raw: torch.Tensor) -> torch.Tensor:
        # bounded scalar
        T = self.raw_temp
        return self.w_min + (self.w_max - self.w_min) * torch.sigmoid(w_raw / T)

    def height(self, h_raw: torch.Tensor) -> torch.Tensor:
        if self.fixed_height is not None:
            return torch.tensor(float(self.fixed_height), device=h_raw.device, dtype=h_raw.dtype)
        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(h_raw)

    def _soft_extrema_levels(self, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Smooth estimates of low/high levels of Q.
        Q: (N,)
        returns (Q_lo, Q_hi) scalars
        """
        Qv = Q.flatten()  # (N,)
        # smooth "max-like"
        w_hi = torch.softmax(Qv / (self.q_hi_temp + self.eps), dim=0)
        Q_hi = (w_hi * Qv).sum()

        # smooth "min-like"
        w_lo = torch.softmax(-Qv / (self.q_lo_temp + self.eps), dim=0)
        Q_lo = (w_lo * Qv).sum()

        return Q_lo, Q_hi

    def Q_threshold_adaptive(self, Q_used: torch.Tensor, w_raw: torch.Tensor) -> torch.Tensor:
        """
        Q_thr = Q_hi - (Q_hi - Q_lo) * f, where Q_lo/Q_hi are soft-quantiles of Q_used.
        f = sigmoid(w_raw/raw_temp)
        """
        f = torch.sigmoid(w_raw / self.raw_temp)  # scalar in (0,1)

        Q_lo = self._soft_quantile(Q_used, self.p_lo)
        Q_hi = self._soft_quantile(Q_used, self.p_hi)

        # ensure non-degenerate / positive range (still smooth)
        Q_range = (Q_hi - Q_lo).clamp_min(1e-4)

        Q_thr = Q_hi - Q_range * f
        return Q_thr
    
    def _soft_quantile(self, Q: torch.Tensor, p: float) -> torch.Tensor:
        """
        Differentiable approximation of quantile via a smooth CDF and fixed Newton iterations.
        Q: (N,)
        p: desired quantile in (0,1)
        returns scalar q such that mean(sigmoid((q - Q)/s)) ~= p
        """
        Qv = Q.flatten()

        # initial guess: mean (stable)
        q = Qv.mean()

        s = self.q_smooth + self.eps
        target = torch.tensor(float(p), device=Qv.device, dtype=Qv.dtype)

        for _ in range(self.q_newton_iters):
            z = (q - Qv) / s
            sig = torch.sigmoid(z)  # (N,)
            f = sig.mean() - target

            # derivative df/dq = mean(sig*(1-sig))/s
            df = (sig * (1.0 - sig)).mean() / s
            df = df.clamp_min(self.q_deriv_eps)

            q = q - f / df

        return q

    def metric_matrices(
        self,
        theta: torch.Tensor,
        a_raw: torch.Tensor,
        a_min: float = 0.5,
        a_max: float = 2.0,
    ) -> torch.Tensor:
        if theta.ndim != 1 or a_raw.ndim != 1 or theta.shape != a_raw.shape:
            raise ValueError(f"metric_matrices expects theta and a_raw of shape (S,), got {theta.shape}, {a_raw.shape}")

        S = theta.shape[0]
        t = torch.tanh(a_raw)
        a = 0.5 * (a_max - a_min) * t + 0.5 * (a_max + a_min)

        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack(
            [torch.stack([c, -s], -1), torch.stack([s, c], -1)],
            -2,
        )  # (S,2,2)

        D = torch.zeros((S, 2, 2), device=R.device, dtype=R.dtype)
        D[:, 0, 0] = a
        D[:, 1, 1] = 1.0 / (a + self.eps)

        return R.transpose(1, 2) @ D @ R  # (S,2,2)

    # -------------------- boundary band (functional) --------------------
    def apply_boundary_band(
        self,
        rho_v: torch.Tensor,
        strength: float = 1.0,
        detach_override: bool = False,
        ref_mode: str = "quantile",
        ref_q: float = 0.99,
    ) -> torch.Tensor:
        if (self.boundary_solid_idx.numel() == 0):
            return rho_v

        rho = rho_v.clone()
        N = rho.numel()

        interior_mask = torch.ones(N, device=rho.device, dtype=torch.bool)
        interior_mask[self.boundary_solid_idx] = False

        vals = rho[interior_mask] if interior_mask.any() else rho
        if detach_override:
            vals = vals.detach()

        if ref_mode == "max":
            ref_level = vals.max()
        elif ref_mode == "quantile":
            ref_level = torch.quantile(vals, ref_q)
        else:
            raise ValueError(f"Unknown ref_mode={ref_mode}. Use 'max' or 'quantile'.")

        ref_level = ref_level.clamp(0.0, 1.0)

        rho[self.boundary_solid_idx] = ref_level.detach() if detach_override else ref_level
        return rho.clamp(0.0, 1.0)

    # -------------------- periodic helpers --------------------
    def _wrap_duv_points_to_seeds(self, diff: torch.Tensor, points_face_id: torch.Tensor | None) -> torch.Tensor:
        """
        diff: (N,S,2) = points - seeds
        points_face_id: (N,) long
        """
        if points_face_id is None:
            return diff
        if points_face_id.dtype != torch.long:
            points_face_id = points_face_id.to(torch.long)

        uper = self.face_u_periodic[points_face_id].to(diff.dtype)  # (N,)
        vper = self.face_v_periodic[points_face_id].to(diff.dtype)  # (N,)

        du = diff[..., 0]
        dv = diff[..., 1]
        du = du - torch.round(du) * uper[:, None]
        dv = dv - torch.round(dv) * vper[:, None]
        diff[..., 0] = du
        diff[..., 1] = dv
        return diff

    def _pairwise_uv_dirs(self, seeds: torch.Tensor) -> torch.Tensor:
        """
        seeds: (S,2)
        returns unit tangent directions between each seed pair: (S,S,2)
        """
        v = seeds.unsqueeze(0) - seeds.unsqueeze(1)  # (S,S,2)
        same_face = (self.seed_face_id[:, None] == self.seed_face_id[None, :])

        uper_face = self.face_u_periodic[self.seed_face_id]  # (S,)
        vper_face = self.face_v_periodic[self.seed_face_id]  # (S,)

        uper_pair = (uper_face[:, None] & uper_face[None, :] & same_face)
        vper_pair = (vper_face[:, None] & vper_face[None, :] & same_face)

        du = v[..., 0]
        dv = v[..., 1]
        du = du - torch.round(du) * uper_pair.to(du.dtype)
        dv = dv - torch.round(dv) * vper_pair.to(dv.dtype)
        v[..., 0] = du
        v[..., 1] = dv

        t = torch.stack([-v[..., 1], v[..., 0]], dim=-1)  # perpendicular in UV
        n = torch.norm(v, dim=-1, keepdim=True) + self.eps
        return t / n

    # -------------------- fiber math (UV) --------------------
    def _soft_pair_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        weights: (N,S)
        returns pi: (N,S,S) normalized upper-triangular pair weights
        """
        N, S = weights.shape
        pair = weights.unsqueeze(2) * weights.unsqueeze(1)  # (N,S,S)
        mask = torch.triu(torch.ones(S, S, device=weights.device, dtype=weights.dtype), diagonal=1)
        pair = pair * mask
        denom = pair.sum(dim=(1, 2), keepdim=True) + self.eps
        return pair / denom

    def _blended_uv_fiber(self, weights: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        """
        weights: (N,S)
        seeds: (S,2)
        returns blended unit fiber direction in UV: (N,2)
        """
        pi = self._soft_pair_weights(weights)  # (N,S,S)
        t_ij = self._pairwise_uv_dirs(seeds)  # (S,S,2)
        T = (pi.unsqueeze(-1) * t_ij.unsqueeze(0)).sum(dim=(1, 2))  # (N,2)
        return T / (torch.norm(T, dim=1, keepdim=True) + self.eps)

    # -------------------- new: differentiable pair-gated Q --------------------
    def _map_raw_to_range(self, x_raw: torch.Tensor, lo: float, hi: float, temp: float = 1.0) -> torch.Tensor:
        """
        Smoothly map raw scalar to [lo,hi]. temp controls slope (1.0 is fine).
        """
        return lo + (hi - lo) * torch.sigmoid(x_raw / temp)

    def map_to_3d(self,t_uv, Xu, Xv, eps=1e-8):
        # t_uv: (N,2), Xu/Xv: (N,3)
        T = t_uv[:, 0:1] * Xu + t_uv[:, 1:2] * Xv
        return F.normalize(T, eps=eps)
    def _pair_gated_Q(
        self,
        w_soft: torch.Tensor,             # (N,S)
        gap_thr: torch.Tensor,            # scalar
        big_thr: torch.Tensor,            # scalar
        alpha: torch.Tensor,              # scalar > 0
        eta: torch.Tensor,                # scalar > 0
    ) -> torch.Tensor:
        """
        Q_pair(x) = sum_{i<j} g_gap(i,j,x) * g_big(i,j,x) * (w_i w_j)

        g_gap(i,j,x) = sigmoid((gap_thr - |w_i - w_j|)/alpha)
        g_big(i,j,x) = sigmoid((w_i - big_thr)/eta) * sigmoid((w_j - big_thr)/eta)

        Returns: (N,)
        """
        N, S = w_soft.shape
        w_i = w_soft.unsqueeze(2)  # (N,S,1)
        w_j = w_soft.unsqueeze(1)  # (N,1,S)

        # upper-triangular mask i<j
        tri = torch.triu(torch.ones(S, S, device=w_soft.device, dtype=w_soft.dtype), diagonal=1)

        # pair weight
        pij = (w_i * w_j) * tri  # (N,S,S)

        # gap gate (close competition)
        gap = torch.abs(w_i - w_j)  # (N,S,S)
        g_gap = torch.sigmoid((gap_thr - gap) / (alpha + self.eps))

        # "both are big" gate
        gi = torch.sigmoid((w_i - big_thr) / (eta + self.eps))
        gj = torch.sigmoid((w_j - big_thr) / (eta + self.eps))
        g_big = gi * gj

        # combine gates
        g = (g_gap * g_big) * tri  # (N,S,S)

        # score
        Q_pair = (g * pij).sum(dim=(1, 2))  # (N,)

        # Optional normalization: keep scale less sensitive to seed count.
        # Denominator is the total pair-mass sum_{i<j} w_i w_j = 0.5*(1 - sum w^2)
        denom = pij.sum(dim=(1, 2)) + self.eps
        Q_pair = Q_pair / denom  # now roughly in [0,1] and comparable across S

        return Q_pair.clamp(0.0, 1.0)

    # -------------------- forward --------------------
    def forward(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        tau: float,
        seeds_raw: torch.Tensor,
        w_raw: torch.Tensor,
        h_raw: torch.Tensor,
        theta: torch.Tensor | None = None,
        a_raw: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
        gap_thr_raw: torch.Tensor | None = None,
        big_thr_raw: torch.Tensor | None = None,
        alpha_raw: torch.Tensor | None = None,
        eta_raw: torch.Tensor | None = None,
    ):
        """
        returns:
          w_soft (N,S), d (N,S), M (S,2,2), seeds (S,2),
          rho (N,), t_uv_raw (N,2), t_uv (N,2), h (scalar tensor), Q_used (N,)
        """
        if points_uv.ndim != 2 or points_uv.shape[1] != 2:
            raise ValueError(f"points_uv must be (N,2), got {tuple(points_uv.shape)}")

        if seeds_raw.shape != (self.n_seeds, 2):
            raise ValueError(f"seeds_raw must be (S,2) with S={self.n_seeds}, got {tuple(seeds_raw.shape)}")

        if not (tau > 0.0):
            raise ValueError(f"tau must be > 0, got {tau}")

        if self.use_anisotropy:
            if theta is None or a_raw is None:
                raise ValueError("use_anisotropy=True requires theta and a_raw.")
            if theta.shape != (self.n_seeds,) or a_raw.shape != (self.n_seeds,):
                raise ValueError(f"theta/a_raw must be (S,) with S={self.n_seeds}, got {theta.shape}, {a_raw.shape}")

        # seeds
        seeds = self.seeds_uv(seeds_raw)  # (S,2)
        S = seeds.shape[0]

        # metric matrices
        if self.use_anisotropy:
            M = self.metric_matrices(theta, a_raw)  # (S,2,2)
        else:
            I = torch.eye(2, device=points_uv.device, dtype=points_uv.dtype)
            M = I.unsqueeze(0).expand(S, 2, 2)

        # distances with periodic wrapping
        diff = points_uv.unsqueeze(1) - seeds.unsqueeze(0)  # (N,S,2)
        diff = self._wrap_duv_points_to_seeds(diff, points_face_id)
        d = torch.einsum("nsi,sij,nsj->ns", diff, M, diff)  # (N,S)

        # optional: block seeds from other faces
        if points_face_id is not None:
            if points_face_id.dtype != torch.long:
                points_face_id = points_face_id.to(torch.long)
            mask = (points_face_id[:, None] != self.seed_face_id[None, :])  # (N,S)
            d = d + mask.to(d.dtype) * 1e6

        # soft assignment weights
        logits = -d / float(tau)
        logits = logits - logits.max(dim=1, keepdim=True).values
        logits = logits.clamp(-80.0, 0.0)
        w_soft = F.softmax(logits, dim=1)  # (N,S)

        # --- edge indicator Q (improved) ---
        if self.use_pair_gated_Q:
            device = w_soft.device
            dtype = w_soft.dtype

            # Use provided raw controls, otherwise fixed defaults (constants on device)
            if gap_thr_raw is None:
                gap_thr = torch.tensor(self.gap_thr_default, device=device, dtype=dtype)
            else:
                gap_thr = self._map_raw_to_range(gap_thr_raw, self.gap_thr_min, self.gap_thr_max, temp=1.0)

            if big_thr_raw is None:
                big_thr = torch.tensor(self.big_thr_default, device=device, dtype=dtype)
            else:
                big_thr = self._map_raw_to_range(big_thr_raw, self.big_thr_min, self.big_thr_max, temp=1.0)

            if alpha_raw is None:
                alpha = torch.tensor(self.alpha_default, device=device, dtype=dtype)
            else:
                alpha = self._map_raw_to_range(alpha_raw, self.alpha_min, self.alpha_max, temp=1.0)

            if eta_raw is None:
                eta = torch.tensor(self.eta_default, device=device, dtype=dtype)
            else:
                eta = self._map_raw_to_range(eta_raw, self.eta_min, self.eta_max, temp=1.0)

            Q_used = self._pair_gated_Q(w_soft, gap_thr=gap_thr, big_thr=big_thr, alpha=alpha, eta=eta)  # (N,)
        else:
            # fallback to original
            Q_used = 1.0 - (w_soft * w_soft).sum(dim=1)  # (N,)

        # rho from width threshold (same mechanism; you may need to re-tune Q_min/Q_max for new Q_used)
        Q_thr = self.Q_threshold_adaptive(Q_used, w_raw)
        rho = torch.sigmoid((Q_used - Q_thr) / self.beta)

        # boundary band
        rho = self.apply_boundary_band(rho)

        # fiber
        t_uv_raw = self._blended_uv_fiber(w_soft, seeds)  # (N,2)
        fiber3d = self.map_to_3d(t_uv_raw, Xu=Xu, Xv=Xv)  # (N,3)
        rho0, gamma = 0.5, 0.05
        m = torch.sigmoid((rho - rho0) / gamma).unsqueeze(1)  # (N,1)
        t_uv = t_uv_raw * m

        # height
        h = self.height(h_raw)  # scalar tensor

        return w_soft, d, M, seeds, rho, t_uv_raw, t_uv, h,Q_used,fiber3d
