import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoronoiDecoder(nn.Module):
    """
    Functional Voronoi decoder with geometric strut thickness centered on Voronoi bisectors.

    Main change:
      - Replace Q-threshold density with a differentiable bisector-band density.

    For each pair (i,j):
        Delta_ij = d_i - d_j
        B_ij = sigmoid((w - |Delta_ij|_smooth) / beta)
        P_ij = soft pair relevance
        S_ij = P_ij * B_ij

    Final density:
        rho = 1 - exp(-alpha_union * sum_{i<j} S_ij)

    This gives width a direct geometric meaning:
        w = half-thickness around the Voronoi bisector.
    """

    def __init__(
        self,
        n_seeds: int,
        eps: float = 1e-8,
        use_anisotropy: bool = True,

        # geometric strut half-width bounds
        w_min: float = 0.005,
        w_max: float = 0.5,

        # density transition sharpness
        beta: float = 0.02,

        # raw parameter temperature for bounded maps
        raw_temp: float = 5.0,

        # union sharpness for combining pair bands
        alpha_union: float = 8.0,

        # height controls
        h_min: float = 0.50,
        h_max: float = 2.00,
        fixed_height: float | None = None,

        # boundary & periodicity
        boundary_solid_idx: torch.Tensor | None = None,
        face_u_periodic: torch.Tensor | None = None,
        face_v_periodic: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,

        # pair-gating controls
        use_pair_gating: bool = True,
        gap_thr_min: float = 0.00,
        gap_thr_max: float = 0.50,
        big_thr_min: float = 0.00,
        big_thr_max: float = 0.60,
        alpha_min: float = 0.01,
        alpha_max: float = 0.20,
        eta_min: float = 0.01,
        eta_max: float = 0.20,

        gap_thr_default: float = 0.15,
        big_thr_default: float = 0.10,
        alpha_default: float = 0.05,
        eta_default: float = 0.05,
        # boundary attachment field
        use_boundary_attachment: bool = True,
        boundary_attach_width: float = 0.03,
        boundary_attach_beta: float = 0.01,
        boundary_attach_alpha: float = 0.35,
    ):
        super().__init__()

        self.n_seeds = int(n_seeds)
        self.eps = float(eps)
        self.use_anisotropy = bool(use_anisotropy)

        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.beta = float(beta)
        self.raw_temp = float(raw_temp)
        self.alpha_union = float(alpha_union)

        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.fixed_height = float(fixed_height) if fixed_height is not None else None

        self.use_pair_gating = bool(use_pair_gating)

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

        self.use_boundary_attachment = bool(use_boundary_attachment)
        self.boundary_attach_width = float(boundary_attach_width)
        self.boundary_attach_beta = float(boundary_attach_beta)
        self.boundary_attach_alpha = float(boundary_attach_alpha)

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
        return seeds_raw

    def width(self, w_raw: torch.Tensor) -> torch.Tensor:
        T = self.raw_temp
        return self.w_min + (self.w_max - self.w_min) * torch.sigmoid(w_raw / T)

    def height(self, h_raw: torch.Tensor | None, ref_tensor: torch.Tensor | None = None) -> torch.Tensor:
        if self.fixed_height is not None:
            if ref_tensor is not None:
                return torch.tensor(float(self.fixed_height), device=ref_tensor.device, dtype=ref_tensor.dtype)
            if h_raw is not None:
                return torch.tensor(float(self.fixed_height), device=h_raw.device, dtype=h_raw.dtype)
            return torch.tensor(float(self.fixed_height))
        if h_raw is None:
            raise ValueError("h_raw must be provided when fixed_height is None")
        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(h_raw)

    def _map_raw_to_range(self, x_raw: torch.Tensor, lo: float, hi: float, temp: float = 1.0) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(x_raw / temp)

    # -------------------- anisotropic metric --------------------

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

        return R.transpose(1, 2) @ D @ R

    # -------------------- periodic helpers --------------------

    def _wrap_duv_points_to_seeds(self, diff: torch.Tensor, points_face_id: torch.Tensor | None) -> torch.Tensor:
        if points_face_id is None:
            return diff
        if points_face_id.dtype != torch.long:
            points_face_id = points_face_id.to(torch.long)

        uper = self.face_u_periodic[points_face_id].to(diff.dtype)
        vper = self.face_v_periodic[points_face_id].to(diff.dtype)

        du = diff[..., 0]
        dv = diff[..., 1]
        du = du - torch.round(du) * uper[:, None]
        dv = dv - torch.round(dv) * vper[:, None]
        diff[..., 0] = du
        diff[..., 1] = dv
        return diff

    def _pairwise_uv_dirs(self, seeds: torch.Tensor) -> torch.Tensor:
        v = seeds.unsqueeze(0) - seeds.unsqueeze(1)  # (S,S,2)
        same_face = (self.seed_face_id[:, None] == self.seed_face_id[None, :])

        uper_face = self.face_u_periodic[self.seed_face_id]
        vper_face = self.face_v_periodic[self.seed_face_id]

        uper_pair = (uper_face[:, None] & uper_face[None, :] & same_face)
        vper_pair = (vper_face[:, None] & vper_face[None, :] & same_face)

        du = v[..., 0]
        dv = v[..., 1]
        du = du - torch.round(du) * uper_pair.to(du.dtype)
        dv = dv - torch.round(dv) * vper_pair.to(dv.dtype)
        v[..., 0] = du
        v[..., 1] = dv

        t = torch.stack([-v[..., 1], v[..., 0]], dim=-1)
        n = torch.norm(v, dim=-1, keepdim=True) + self.eps
        return t / n

    # -------------------- fiber helpers --------------------

    def _soft_pair_weights(self, weights: torch.Tensor) -> torch.Tensor:
        N, S = weights.shape
        pair = weights.unsqueeze(2) * weights.unsqueeze(1)
        mask = torch.triu(torch.ones(S, S, device=weights.device, dtype=weights.dtype), diagonal=1)
        pair = pair * mask
        denom = pair.sum(dim=(1, 2), keepdim=True) + self.eps
        return pair / denom

    def _blended_uv_fiber(self, weights: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        pi = self._soft_pair_weights(weights)
        t_ij = self._pairwise_uv_dirs(seeds)
        T = (pi.unsqueeze(-1) * t_ij.unsqueeze(0)).sum(dim=(1, 2))
        return T / (torch.norm(T, dim=1, keepdim=True) + self.eps)

    def map_to_3d(self, t_uv, Xu, Xv, eps=1e-8):
        T = t_uv[:, 0:1] * Xu + t_uv[:, 1:2] * Xv
        return F.normalize(T, eps=eps)

    # -------------------- boundary band --------------------
    def boundary_attachment_field(
        self,
        points_uv: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        points_face_id: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Smooth boundary attachment field rho_b in [0,1].

        High near the shell boundary, decays smoothly away from it.
        """
        if boundary_uv is None or boundary_uv.numel() == 0:
            return torch.zeros(
                points_uv.shape[0],
                device=points_uv.device,
                dtype=points_uv.dtype,
            )

        # piecewise-smooth nearest-boundary distance
        if boundary_face_id is not None and points_face_id is not None:
            if boundary_face_id.dtype != torch.long:
                boundary_face_id = boundary_face_id.to(torch.long)
            if points_face_id.dtype != torch.long:
                points_face_id = points_face_id.to(torch.long)

            dmat = torch.cdist(points_uv, boundary_uv)  # [Nv, Nb]
            cross_face = points_face_id[:, None] != boundary_face_id[None, :]
            dmat = dmat + cross_face.to(dmat.dtype) * 1e6
            dmin = dmat.amin(dim=1)
        else:
            dmin = torch.cdist(points_uv, boundary_uv).amin(dim=1)

        tb = torch.as_tensor(
            self.boundary_attach_width,
            device=points_uv.device,
            dtype=points_uv.dtype,
        )
        bb = torch.as_tensor(
            self.boundary_attach_beta,
            device=points_uv.device,
            dtype=points_uv.dtype,
        )

        rho_b = torch.sigmoid((tb - dmin) / (bb + self.eps))
        return rho_b.clamp(0.0, 1.0)
    
    def smooth_union(
        self,
        rho_a: torch.Tensor,
        rho_b: torch.Tensor,
        alpha_b: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Smooth union of two density-like fields in [0,1].

        rho = 1 - (1-rho_a)(1-alpha_b*rho_b)
        """
        alpha_b = torch.as_tensor(alpha_b, device=rho_a.device, dtype=rho_a.dtype)
        rho = 1.0 - (1.0 - rho_a) * (1.0 - alpha_b * rho_b)
        return rho.clamp(0.0, 1.0)

  # -------------------- pair gating --------------------
    def _pair_gates(
        self,
        w_soft: torch.Tensor,
        gap_thr: torch.Tensor,
        big_thr: torch.Tensor,
        alpha: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns pair gates g_ij in [0,1], shape (N,S,S), upper triangular only.
        """
        N, S = w_soft.shape
        w_i = w_soft.unsqueeze(2)
        w_j = w_soft.unsqueeze(1)

        tri = torch.triu(torch.ones(S, S, device=w_soft.device, dtype=w_soft.dtype), diagonal=1)

        gap = torch.abs(w_i - w_j)
        g_gap = torch.sigmoid((gap_thr - gap) / (alpha + self.eps))

        gi = torch.sigmoid((w_i - big_thr) / (eta + self.eps))
        gj = torch.sigmoid((w_j - big_thr) / (eta + self.eps))
        g_big = gi * gj

        return (g_gap * g_big) * tri

    # -------------------- new geometric strut density --------------------
    def _bisector_band_density(
        self,
        d: torch.Tensor,                 # (N,S)
        w_soft: torch.Tensor,            # (N,S)
        w_geo: torch.Tensor,             # scalar half-width
        beta: float | torch.Tensor,      # scalar
        gap_thr: torch.Tensor | None = None,
        big_thr: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        eta: torch.Tensor | None = None,
    ):
        """
        Smooth strut density centered on Voronoi bisectors.

        For each pair (i,j):
            Delta_ij = d_i - d_j
            band_ij = sigmoid((w_geo - smooth_abs(Delta_ij)) / beta)
            pair_relevance_ij = w_i w_j [optionally gated]
            s_ij = pair_relevance_ij * band_ij

        Aggregate:
            R = sum_{i<j} s_ij
            rho = 1 - exp(-alpha_union * R)

        Returns:
            rho: (N,)
            pair_strength: (N,S,S)
            band_ij: (N,S,S)
            pair_relevance: (N,S,S)
        """
        N, S = d.shape
        d_i = d.unsqueeze(2)
        d_j = d.unsqueeze(1)

        tri = torch.triu(torch.ones(S, S, device=d.device, dtype=d.dtype), diagonal=1)

        delta = d_i - d_j
        abs_delta = torch.sqrt(delta * delta + self.eps)

        beta_t = torch.as_tensor(beta, device=d.device, dtype=d.dtype)
        band_ij = torch.sigmoid((w_geo - abs_delta) / (beta_t + self.eps)) * tri

        pair_relevance = (w_soft.unsqueeze(2) * w_soft.unsqueeze(1)) * tri

        if self.use_pair_gating and gap_thr is not None and big_thr is not None and alpha is not None and eta is not None:
            gates = self._pair_gates(w_soft, gap_thr, big_thr, alpha, eta)
            pair_relevance = pair_relevance * gates

        pair_strength = pair_relevance * band_ij
        R = pair_strength.sum(dim=(1, 2))

        rho = 1.0 - torch.exp(-self.alpha_union * R)
        rho = rho.clamp(0.0, 1.0)
        band_soft = band_ij.clamp(0.0, 1.0)

        S = band_soft.shape[1]
        eye = torch.eye(S, dtype=torch.bool, device=band_soft.device).unsqueeze(0)  # (1,S,S)

        one_minus = torch.where(
            eye,
            torch.ones_like(band_soft),   # ignore diagonal terms
            1.0 - band_soft
        )

        edge_field = 1.0 - one_minus.prod(dim=2).prod(dim=1)
        edge_field = edge_field.clamp(0.0, 1.0)

        return rho, pair_strength, band_ij, pair_relevance,edge_field

    # -------------------- forward --------------------
    def forward(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        tau: float,
        seeds_raw: torch.Tensor,
        w_raw: torch.Tensor,
        h_raw: torch.Tensor | None,
        theta: torch.Tensor | None = None,
        a_raw: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
        boundary_uv: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        gap_thr_raw: torch.Tensor | None = None,
        big_thr_raw: torch.Tensor | None = None,
        alpha_raw: torch.Tensor | None = None,
        eta_raw: torch.Tensor | None = None,
    ):
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

        seeds = self.seeds_uv(seeds_raw)
        S = seeds.shape[0]

        if self.use_anisotropy:
            M = self.metric_matrices(theta, a_raw)
        else:
            I = torch.eye(2, device=points_uv.device, dtype=points_uv.dtype)
            M = I.unsqueeze(0).expand(S, 2, 2)

        diff = points_uv.unsqueeze(1) - seeds.unsqueeze(0)
        diff = self._wrap_duv_points_to_seeds(diff, points_face_id)
        d2 = torch.einsum("nsi,sij,nsj->ns", diff, M, diff)
        d = torch.sqrt(d2.clamp_min(self.eps))

        if points_face_id is not None:
            if self.seed_face_id is None:
                raise ValueError("points_face_id was provided but self.seed_face_id is None.")
            if points_face_id.dtype != torch.long:
                points_face_id = points_face_id.to(torch.long)
            seed_face_id = self.seed_face_id.to(device=points_face_id.device, dtype=torch.long)
            mask = (points_face_id[:, None] != seed_face_id[None, :])
            d = d + mask.to(d.dtype) * 1e6

        logits = -d / float(tau)
        logits = logits - logits.max(dim=1, keepdim=True).values
        logits = logits.clamp(-80.0, 0.0)
        w_soft = F.softmax(logits, dim=1)

        device = w_soft.device
        dtype = w_soft.dtype

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

        w_geo = self.width(w_raw)

        rho, pair_strength, band_ij, pair_relevance,edge_field = self._bisector_band_density(
            d=d,
            w_soft=w_soft,
            w_geo=w_geo,
            beta=self.beta,
            gap_thr=gap_thr,
            big_thr=big_thr,
            alpha=alpha,
            eta=eta,
        )

        rho_v = rho.clamp(0.0, 1.0)

        if self.use_boundary_attachment:
            rho_b = self.boundary_attachment_field(
                points_uv=points_uv,
                boundary_uv=boundary_uv,
                points_face_id=points_face_id,
                boundary_face_id=boundary_face_id,
            )
            rho = self.smooth_union(
                rho_a=rho_v,
                rho_b=rho_b,
                alpha_b=self.boundary_attach_alpha,
            )
        else:
            rho_b = torch.zeros_like(rho_v)
            rho = rho_v

        rho = rho.clamp(0.0, 1.0)

        t_uv_raw = self._blended_uv_fiber(w_soft, seeds)
        rho0, gamma = 0.5, 0.05
        m = torch.sigmoid((rho - rho0) / gamma).unsqueeze(1)
        t_uv = t_uv_raw * m
        fiber3d = self.map_to_3d(t_uv, Xu=Xu, Xv=Xv)

        h = self.height(h_raw, ref_tensor=points_uv)

        return (
            w_soft,
            d,
            M,
            seeds,
            rho,
            t_uv_raw,
            t_uv,
            h,
            w_geo,
            fiber3d,
            pair_strength,
            band_ij,
            pair_relevance,
            rho_v,
            rho_b,
            edge_field,
        )