import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoronoiDecoder(nn.Module):
    """
    Fully functional Voronoi decoder.

    This module contains no trainable parameters.
    All learnable quantities are predicted externally by PPNet and passed in
    through evaluate_at_uv(...) / forward(...).

    PPNet-predicted learnable controls can include:
      - seeds_raw
      - w_raw
      - h_raw
      - theta, a_raw
      - gap_thr_raw, big_thr_raw, alpha_raw, eta_raw
      - boundary_width_raw, boundary_alpha_raw, boundary_beta_raw
      - seed_gates
    """

    def __init__(
        self,
        n_seeds: int,
        eps: float = 1e-8,
        use_Metric_anisotropy: bool = True,

        # geometric strut half-width lower bound
        w_min: float = 0.005,

        # density transition sharpness
        beta: float = 0.02,
        junction_beta_scale: float = 1.0,
        junction_width_bonus: float = 0.15,

        # effective-number boost for multi-seed zones
        junction_keff_lambda: float = 0.050,
        junction_keff_k0: float = 3.0,
        junction_keff_s: float = 0.35,

        # explicit triple-overlap junction term
        junction_triple_lambda: float = 0.15,
        junction_triple_power: float = 1.5,

        # raw parameter temperature for bounded maps
        raw_temp: float = 5.0,

        # union sharpness for combining pair bands
        alpha_union: float = 12.0,

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

        # defaults used only when PPNet does not provide the corresponding raw values
        gap_thr_default: float = 0.15,
        big_thr_default: float = 0.10,
        alpha_default: float = 0.05,
        eta_default: float = 0.05,

        # boundary attachment field
        use_boundary_attachment: bool = False,

        # keep these on comparable scales
        boundary_attach_width: float = 2e-5,
        boundary_attach_beta: float = 1e-5,
        boundary_attach_alpha: float = 0.35,

        boundary_attach_width_min: float = 5e-6,
        boundary_attach_width_max: float = 5e-5,

        boundary_attach_alpha_min: float = 0.05,
        boundary_attach_alpha_max: float = 1.00,

        boundary_attach_beta_min: float = 1e-6,
        boundary_attach_beta_max: float = 1e-4,

        # robust boundary-distance evaluation
        boundary_knn_k: int = 8,
        boundary_softmin_tau: float = 2e-3,
        boundary_spacing_blend: float = 0.5,
    ):
        super().__init__()

        self.n_seeds = int(n_seeds)
        self.eps = float(eps)
        self.use_Metric_anisotropy = bool(use_Metric_anisotropy)

        self.w_min = float(w_min)
        self.beta = float(beta)
        self.junction_beta_scale = float(junction_beta_scale)
        self.junction_width_bonus = float(junction_width_bonus)

        self.junction_keff_lambda = float(junction_keff_lambda)
        self.junction_keff_k0 = float(junction_keff_k0)
        self.junction_keff_s = float(junction_keff_s)

        self.junction_triple_lambda = float(junction_triple_lambda)
        self.junction_triple_power = float(junction_triple_power)

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

        self.boundary_attach_width_min = float(boundary_attach_width_min)
        self.boundary_attach_width_max = float(boundary_attach_width_max)
        self.boundary_attach_alpha_min = float(boundary_attach_alpha_min)
        self.boundary_attach_alpha_max = float(boundary_attach_alpha_max)
        self.boundary_attach_beta_min = float(boundary_attach_beta_min)
        self.boundary_attach_beta_max = float(boundary_attach_beta_max)
        self.boundary_knn_k = int(boundary_knn_k)
        self.boundary_softmin_tau = float(boundary_softmin_tau)
        self.boundary_spacing_blend = float(boundary_spacing_blend)

        if not (self.boundary_attach_width_min < self.boundary_attach_width_max):
            raise ValueError(
                f"boundary_attach_width_min must be < boundary_attach_width_max, got "
                f"{self.boundary_attach_width_min} and {self.boundary_attach_width_max}"
            )
        if not (self.boundary_attach_alpha_min < self.boundary_attach_alpha_max):
            raise ValueError(
                f"boundary_attach_alpha_min must be < boundary_attach_alpha_max, got "
                f"{self.boundary_attach_alpha_min} and {self.boundary_attach_alpha_max}"
            )
        if not (self.boundary_attach_beta_min < self.boundary_attach_beta_max):
            raise ValueError(
                f"boundary_attach_beta_min must be < boundary_attach_beta_max, got "
                f"{self.boundary_attach_beta_min} and {self.boundary_attach_beta_max}"
            )
        if self.boundary_knn_k < 1:
            raise ValueError(f"boundary_knn_k must be >= 1, got {self.boundary_knn_k}")
        if self.boundary_softmin_tau <= 0:
            raise ValueError(f"boundary_softmin_tau must be > 0, got {self.boundary_softmin_tau}")
        if self.boundary_spacing_blend < 0:
            raise ValueError(f"boundary_spacing_blend must be >= 0, got {self.boundary_spacing_blend}")
        if self.junction_triple_power <= 0:
            raise ValueError(f"junction_triple_power must be > 0, got {self.junction_triple_power}")

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

        self.register_buffer(
            "boundary_attach_width_fixed",
            torch.tensor(float(boundary_attach_width), dtype=torch.float32),
        )
        self.register_buffer(
            "boundary_attach_alpha_fixed",
            torch.tensor(float(boundary_attach_alpha), dtype=torch.float32),
        )
        self.register_buffer(
            "boundary_attach_beta_fixed",
            torch.tensor(float(boundary_attach_beta), dtype=torch.float32),
        )

    # -------------------- parameter maps --------------------

    def seeds_uv(self, seeds_raw: torch.Tensor) -> torch.Tensor:
        return seeds_raw

    def _pairwise_seed_dist(self, seeds: torch.Tensor) -> torch.Tensor:
        v = seeds.unsqueeze(0) - seeds.unsqueeze(1)
        same_face = self.seed_face_id[:, None] == self.seed_face_id[None, :]

        uper_face = self.face_u_periodic[self.seed_face_id]
        vper_face = self.face_v_periodic[self.seed_face_id]

        uper_pair = uper_face[:, None] & uper_face[None, :] & same_face
        vper_pair = vper_face[:, None] & vper_face[None, :] & same_face

        du = v[..., 0]
        dv = v[..., 1]

        du = du - torch.round(du) * uper_pair.to(du.dtype)
        dv = dv - torch.round(dv) * vper_pair.to(dv.dtype)

        v[..., 0] = du
        v[..., 1] = dv
        return torch.norm(v, dim=-1)

    def width(self, w_raw: torch.Tensor, seeds: torch.Tensor | None = None) -> torch.Tensor:
        T = self.raw_temp
        if w_raw.ndim != 2 or w_raw.shape[0] != w_raw.shape[1]:
            raise ValueError(f"w_raw must be square (S,S), got {tuple(w_raw.shape)}")
        if seeds is None:
            raise ValueError("seeds must be provided when w_raw is pairwise")
        if seeds.shape[0] != w_raw.shape[0]:
            raise ValueError(
                f"pairwise w_raw expects seeds with matching S, got {tuple(seeds.shape)} and {tuple(w_raw.shape)}"
            )

        pair_dist = self._pairwise_seed_dist(seeds).to(device=w_raw.device, dtype=w_raw.dtype)
        w_max_pair = (0.8 * pair_dist).clamp_min(self.w_min + self.eps)
        w_geo = self.w_min + (w_max_pair - self.w_min) * torch.sigmoid(w_raw / T)
        return 0.5 * (w_geo + w_geo.transpose(0, 1))

    def height(
        self,
        h_raw: torch.Tensor | None,
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fixed_height is not None:
            if ref_tensor is not None:
                return torch.tensor(
                    float(self.fixed_height),
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                )
            if h_raw is not None:
                return torch.tensor(
                    float(self.fixed_height),
                    device=h_raw.device,
                    dtype=h_raw.dtype,
                )
            return torch.tensor(float(self.fixed_height))

        if h_raw is None:
            raise ValueError("h_raw must be provided when fixed_height is None")

        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(h_raw)

    def _map_raw_to_range(
        self,
        x_raw: torch.Tensor,
        lo: float,
        hi: float,
        temp: float = 1.0,
    ) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(x_raw / temp)

    def raw_from_bounded_value(
        self,
        value: float,
        lo: float,
        hi: float,
        temp: float = 1.0,
    ) -> torch.Tensor:
        denom = max(hi - lo, self.eps)
        x = (value - lo) / denom
        x = min(max(x, 1e-6), 1.0 - 1e-6)
        raw = temp * math.log(x / (1.0 - x))
        return torch.tensor(raw, dtype=torch.float32)

    # -------------------- boundary control getters --------------------

    def boundary_width(
        self,
        ref_tensor: torch.Tensor,
        boundary_width_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_width_raw is None:
            return self.boundary_attach_width_fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_width_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return self._map_raw_to_range(
            raw,
            self.boundary_attach_width_min,
            self.boundary_attach_width_max,
            temp=1.0,
        )

    def boundary_alpha(
        self,
        ref_tensor: torch.Tensor,
        boundary_alpha_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_alpha_raw is None:
            return self.boundary_attach_alpha_fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_alpha_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return self._map_raw_to_range(
            raw,
            self.boundary_attach_alpha_min,
            self.boundary_attach_alpha_max,
            temp=1.0,
        )

    def boundary_beta(
        self,
        ref_tensor: torch.Tensor,
        boundary_beta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_beta_raw is None:
            return self.boundary_attach_beta_fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_beta_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return self._map_raw_to_range(
            raw,
            self.boundary_attach_beta_min,
            self.boundary_attach_beta_max,
            temp=1.0,
        )

    # -------------------- anisotropic metric --------------------

    def metric_matrices(
        self,
        theta: torch.Tensor,
        a_raw: torch.Tensor,
        a_min: float = 0.5,
        a_max: float = 2.0,
    ) -> torch.Tensor:
        if theta.ndim != 1 or a_raw.ndim != 1 or theta.shape != a_raw.shape:
            raise ValueError(
                f"metric_matrices expects theta and a_raw of shape (S,), got {theta.shape}, {a_raw.shape}"
            )

        S = theta.shape[0]
        t = torch.tanh(a_raw)
        a = 0.5 * (a_max - a_min) * t + 0.5 * (a_max + a_min)

        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack(
            [torch.stack([c, -s], -1), torch.stack([s, c], -1)],
            -2,
        )

        D = torch.zeros((S, 2, 2), device=R.device, dtype=R.dtype)
        D[:, 0, 0] = a
        D[:, 1, 1] = 1.0 / (a + self.eps)

        return R.transpose(1, 2) @ D @ R

    # -------------------- periodic helpers --------------------

    def _wrap_duv_points_to_seeds(
        self,
        diff: torch.Tensor,
        points_face_id: torch.Tensor | None,
    ) -> torch.Tensor:
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
        v = seeds.unsqueeze(0) - seeds.unsqueeze(1)
        same_face = self.seed_face_id[:, None] == self.seed_face_id[None, :]

        uper_face = self.face_u_periodic[self.seed_face_id]
        vper_face = self.face_v_periodic[self.seed_face_id]

        uper_pair = uper_face[:, None] & uper_face[None, :] & same_face
        vper_pair = vper_face[:, None] & vper_face[None, :] & same_face

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
        mask = torch.triu(
            torch.ones(S, S, device=weights.device, dtype=weights.dtype),
            diagonal=1,
        )
        pair = pair * mask
        denom = pair.sum(dim=(1, 2), keepdim=True) + self.eps
        return pair / denom

    def _blended_uv_fiber(self, weights: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        pi = self._soft_pair_weights(weights)
        t_ij = self._pairwise_uv_dirs(seeds)
        T = (pi.unsqueeze(-1) * t_ij.unsqueeze(0)).sum(dim=(1, 2))
        return T / (torch.norm(T, dim=1, keepdim=True) + self.eps)

    def map_to_3d(self, t_uv: torch.Tensor, Xu: torch.Tensor, Xv: torch.Tensor, eps: float = 1e-8):
        T = t_uv[:, 0:1] * Xu + t_uv[:, 1:2] * Xv
        return F.normalize(T, dim=1, eps=eps)

    # -------------------- boundary band --------------------

    def boundary_attachment_field(
        self,
        points_uv: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        points_face_id: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        boundary_width_raw: torch.Tensor | None = None,
        boundary_beta_raw: torch.Tensor | None = None,
        alpha_union: float = 8.0,
    ) -> torch.Tensor:
        if boundary_uv is None or boundary_uv.numel() == 0:
            return torch.zeros(
                points_uv.shape[0],
                device=points_uv.device,
                dtype=points_uv.dtype,
            )

        dmat = torch.cdist(points_uv, boundary_uv)
        if boundary_face_id is not None and points_face_id is not None:
            if boundary_face_id.dtype != torch.long:
                boundary_face_id = boundary_face_id.to(torch.long)
            if points_face_id.dtype != torch.long:
                points_face_id = points_face_id.to(torch.long)

            cross_face = points_face_id[:, None] != boundary_face_id[None, :]
            dmat = dmat + cross_face.to(dmat.dtype) * 1e6

        k = min(self.boundary_knn_k, int(dmat.shape[1]))
        d_knn = torch.topk(dmat, k=k, dim=1, largest=False).values

        tau = torch.as_tensor(self.boundary_softmin_tau, device=dmat.device, dtype=dmat.dtype)
        dmin = -tau * torch.logsumexp(-d_knn / (tau + self.eps), dim=1) + tau * math.log(k)

        tb = self.boundary_width(points_uv, boundary_width_raw=boundary_width_raw)
        bb = self.boundary_beta(points_uv, boundary_beta_raw=boundary_beta_raw)
        if k > 1 and self.boundary_spacing_blend > 0.0 and boundary_uv.shape[0] > 1:
            b2b = torch.cdist(boundary_uv, boundary_uv)
            big = torch.eye(boundary_uv.shape[0], device=b2b.device, dtype=b2b.dtype) * 1e6
            b2b = b2b + big
            h_boundary = b2b.min(dim=1).values.median()
            bb = bb + self.boundary_spacing_blend * h_boundary

        rho_b_raw = torch.sigmoid((tb - dmin) / (bb + self.eps))
        norm = torch.sigmoid(tb / (bb + self.eps))
        rho_b_norm = (rho_b_raw / (norm + self.eps)).clamp(0.0, 1.0)

        rho_b = 1.0 - torch.exp(-alpha_union * rho_b_norm)
        return rho_b.clamp(0.0, 1.0)

    def smooth_union(
        self,
        rho_a: torch.Tensor,
        rho_b: torch.Tensor,
        alpha_b: float | torch.Tensor,
    ) -> torch.Tensor:
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
        N, S = w_soft.shape

        w_i = w_soft.unsqueeze(2)
        w_j = w_soft.unsqueeze(1)

        tri = torch.triu(
            torch.ones(S, S, device=w_soft.device, dtype=w_soft.dtype),
            diagonal=1,
        )

        gap = torch.abs(w_i - w_j)
        g_gap = torch.sigmoid((gap_thr - gap) / (alpha + self.eps))

        gi = torch.sigmoid((w_i - big_thr) / (eta + self.eps))
        gj = torch.sigmoid((w_j - big_thr) / (eta + self.eps))
        g_big = gi * gj

        return (g_gap * g_big) * tri

    # -------------------- higher-order helpers --------------------

    def _triple_junction_score(self, w_soft: torch.Tensor) -> torch.Tensor:
        """
        Returns a per-point scalar measuring simultaneous 3-seed overlap:
            sum_{i<j<k} (w_i w_j w_k)^p
        where p = junction_triple_power.
        """
        N, S = w_soft.shape
        if S < 3:
            return torch.zeros(N, device=w_soft.device, dtype=w_soft.dtype)

        wi = w_soft.unsqueeze(2).unsqueeze(3)  # (N,S,1,1)
        wj = w_soft.unsqueeze(1).unsqueeze(3)  # (N,1,S,1)
        wk = w_soft.unsqueeze(1).unsqueeze(2)  # (N,1,1,S)

        triple = wi * wj * wk  # (N,S,S,S)
        if self.junction_triple_power != 1.0:
            triple = triple.pow(self.junction_triple_power)

        idx = torch.arange(S, device=w_soft.device)
        mask = (
            (idx.view(S, 1, 1) < idx.view(1, S, 1)) &
            (idx.view(1, S, 1) < idx.view(1, 1, S))
        ).to(w_soft.dtype)

        return (triple * mask.unsqueeze(0)).sum(dim=(1, 2, 3))

    # -------------------- bisector band density --------------------

    def _bisector_band_density(
        self,
        d: torch.Tensor,
        w_soft: torch.Tensor,
        w_geo: torch.Tensor,
        beta: float | torch.Tensor,
        gap_thr: torch.Tensor | None = None,
        big_thr: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        eta: torch.Tensor | None = None,
        seed_gates: torch.Tensor | None = None,
    ):
        N, S = d.shape

        d_i = d.unsqueeze(2)
        d_j = d.unsqueeze(1)

        tri = torch.triu(
            torch.ones(S, S, device=d.device, dtype=d.dtype),
            diagonal=1,
        )

        delta = d_i - d_j
        abs_delta = torch.sqrt(delta * delta + self.eps)

        ambiguity = (1.0 - w_soft.pow(2).sum(dim=1)).clamp(0.0, 1.0)

        beta_t = torch.as_tensor(beta, device=d.device, dtype=d.dtype)
        beta_eff = beta_t * (
            1.0 + self.junction_beta_scale * ambiguity.unsqueeze(1).unsqueeze(2)
        )
        w_geo_eff = w_geo * (
            1.0 + self.junction_width_bonus * ambiguity.unsqueeze(1).unsqueeze(2)
        )

        band_raw = torch.sigmoid((w_geo_eff - abs_delta) / (beta_eff + self.eps))
        band_peak = torch.sigmoid(w_geo_eff / (beta_eff + self.eps))
        band_ij = (band_raw / (band_peak + self.eps)).clamp(0.0, 1.0) * tri

        # pair selectivity based on actual seed participation
        pair_prod = w_soft.unsqueeze(2) * w_soft.unsqueeze(1)

        # effective number of active seeds
        sum_w2 = w_soft.pow(2).sum(dim=1).clamp_min(self.eps)
        k_eff = 1.0 / sum_w2
        junction_mult = 1.0 + self.junction_keff_lambda * torch.sigmoid(
            (k_eff - self.junction_keff_k0) / (self.junction_keff_s + self.eps)
        )

        pair_relevance = (
            ambiguity.unsqueeze(1).unsqueeze(2)
            * pair_prod
            * junction_mult.unsqueeze(1).unsqueeze(2)
            * tri
        )

        gate_pair = None
        if seed_gates is not None:
            if seed_gates.ndim != 1 or seed_gates.shape[0] != S:
                raise ValueError(
                    f"seed_gates must have shape ({S},), got {tuple(seed_gates.shape)}"
                )
            g = seed_gates.to(device=d.device, dtype=d.dtype).clamp_min(self.eps)
            gate_pair = (g.unsqueeze(1) * g.unsqueeze(0)) * tri
            pair_relevance = pair_relevance * gate_pair.unsqueeze(0)

        if (
            self.use_pair_gating
            and gap_thr is not None
            and big_thr is not None
            and alpha is not None
            and eta is not None
        ):
            gates = self._pair_gates(w_soft, gap_thr, big_thr, alpha, eta)
            pair_relevance = pair_relevance * gates

        pair_strength = pair_relevance * band_ij
        R_pair = pair_strength.sum(dim=(1, 2))

        # explicit higher-order junction support
        R_junction = self._triple_junction_score(w_soft)
        R = R_pair + self.junction_triple_lambda * R_junction

        band_soft = band_ij
        if gate_pair is not None:
            band_soft = band_soft * gate_pair.unsqueeze(0)
        band_soft = band_soft.clamp(0.0, 1.0)

        eye = torch.eye(S, dtype=torch.bool, device=band_soft.device).unsqueeze(0)
        one_minus = torch.where(
            eye,
            torch.ones_like(band_soft),
            1.0 - band_soft,
        )

        rho = 1.0 - torch.exp(-self.alpha_union * R)
        rho = rho.clamp(0.0, 1.0)

        edge_field = 1.0 - one_minus.prod(dim=2).prod(dim=1)
        edge_field = edge_field.clamp(0.0, 1.0)

        return rho, pair_strength, band_ij, pair_relevance, edge_field

    # -------------------- validation --------------------

    def _validate_inputs(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        tau: float,
        seeds_raw: torch.Tensor,
        w_raw: torch.Tensor,
        theta: torch.Tensor | None,
        a_raw: torch.Tensor | None,
    ) -> None:
        if points_uv.ndim != 2 or points_uv.shape[1] != 2:
            raise ValueError(f"points_uv must be (N,2), got {tuple(points_uv.shape)}")
        if Xu.ndim != 2 or Xu.shape[1] != 3:
            raise ValueError(f"Xu must be (N,3), got {tuple(Xu.shape)}")
        if Xv.ndim != 2 or Xv.shape[1] != 3:
            raise ValueError(f"Xv must be (N,3), got {tuple(Xv.shape)}")
        if Xu.shape[0] != points_uv.shape[0] or Xv.shape[0] != points_uv.shape[0]:
            raise ValueError("points_uv, Xu, and Xv must have the same first dimension")
        if seeds_raw.shape != (self.n_seeds, 2):
            raise ValueError(
                f"seeds_raw must be (S,2) with S={self.n_seeds}, got {tuple(seeds_raw.shape)}"
            )
        if w_raw.shape != (self.n_seeds, self.n_seeds):
            raise ValueError(
                f"w_raw must be (S,S) with S={self.n_seeds}, got {tuple(w_raw.shape)}"
            )
        if not (tau > 0.0):
            raise ValueError(f"tau must be > 0, got {tau}")
        if self.use_Metric_anisotropy:
            if theta is None or a_raw is None:
                raise ValueError("use_Metric_anisotropy=True requires theta and a_raw.")
            if theta.shape != (self.n_seeds,) or a_raw.shape != (self.n_seeds,):
                raise ValueError(
                    f"theta/a_raw must be (S,) with S={self.n_seeds}, got {theta.shape}, {a_raw.shape}"
                )

    # -------------------- field evaluation --------------------

    def evaluate_at_uv(
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
        boundary_width_raw: torch.Tensor | None = None,
        boundary_alpha_raw: torch.Tensor | None = None,
        boundary_beta_raw: torch.Tensor | None = None,
        seed_gates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(
            points_uv=points_uv,
            Xu=Xu,
            Xv=Xv,
            tau=tau,
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            theta=theta,
            a_raw=a_raw,
        )

        seeds = self.seeds_uv(seeds_raw)
        S = seeds.shape[0]

        if self.use_Metric_anisotropy:
            M = self.metric_matrices(theta, a_raw)
        else:
            I = torch.eye(2, device=points_uv.device, dtype=points_uv.dtype)
            M = I.unsqueeze(0).expand(S, 2, 2)

        diff = points_uv.unsqueeze(1) - seeds.unsqueeze(0)
        diff = self._wrap_duv_points_to_seeds(diff, points_face_id)

        d2 = torch.einsum("nsi,sij,nsj->ns", diff, M, diff)
        d = torch.sqrt(d2.clamp_min(self.eps))

        if points_face_id is not None:
            if points_face_id.dtype != torch.long:
                points_face_id = points_face_id.to(torch.long)

            seed_face_id = self.seed_face_id.to(
                device=points_face_id.device,
                dtype=torch.long,
            )

            cross_face_mask = points_face_id[:, None] != seed_face_id[None, :]
            d = d + cross_face_mask.to(d.dtype) * 1e6

        logits = -d / tau

        gates = None
        if seed_gates is not None:
            if seed_gates.ndim != 1 or seed_gates.shape[0] != S:
                raise ValueError(
                    f"seed_gates must have shape ({S},), got {tuple(seed_gates.shape)}"
                )
            gates = seed_gates.to(device=d.device, dtype=d.dtype).clamp_min(1e-8)
            logits = logits + torch.log(gates).unsqueeze(0)

        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits = logits.clamp(min=-80.0, max=0.0)
        w_soft = torch.softmax(logits, dim=-1)

        device = w_soft.device
        dtype = w_soft.dtype

        if gap_thr_raw is None:
            gap_thr = torch.tensor(self.gap_thr_default, device=device, dtype=dtype)
        else:
            gap_thr = self._map_raw_to_range(
                gap_thr_raw, self.gap_thr_min, self.gap_thr_max, temp=1.0
            )

        if big_thr_raw is None:
            big_thr = torch.tensor(self.big_thr_default, device=device, dtype=dtype)
        else:
            big_thr = self._map_raw_to_range(
                big_thr_raw, self.big_thr_min, self.big_thr_max, temp=1.0
            )

        if alpha_raw is None:
            alpha = torch.tensor(self.alpha_default, device=device, dtype=dtype)
        else:
            alpha = self._map_raw_to_range(
                alpha_raw, self.alpha_min, self.alpha_max, temp=1.0
            )

        if eta_raw is None:
            eta = torch.tensor(self.eta_default, device=device, dtype=dtype)
        else:
            eta = self._map_raw_to_range(
                eta_raw, self.eta_min, self.eta_max, temp=1.0
            )

        w_geo = self.width(w_raw, seeds=seeds)

        rho_v, pair_strength, band_ij, pair_relevance, edge_field = self._bisector_band_density(
            d=d,
            w_soft=w_soft,
            w_geo=w_geo,
            beta=self.beta,
            gap_thr=gap_thr,
            big_thr=big_thr,
            alpha=alpha,
            eta=eta,
            seed_gates=gates,
        )

        if self.use_boundary_attachment:
            rho_b = self.boundary_attachment_field(
                points_uv=points_uv,
                boundary_uv=boundary_uv,
                points_face_id=points_face_id,
                boundary_face_id=boundary_face_id,
                boundary_width_raw=boundary_width_raw,
                boundary_beta_raw=boundary_beta_raw,
            )
            alpha_b = self.boundary_alpha(
                points_uv,
                boundary_alpha_raw=boundary_alpha_raw,
            )
            rho = self.smooth_union(
                rho_a=rho_v,
                rho_b=rho_b,
                alpha_b=alpha_b,
            )
        else:
            rho_b = torch.zeros_like(rho_v)
            alpha_b = torch.zeros((), device=points_uv.device, dtype=points_uv.dtype)
            rho = rho_v

        rho = rho.clamp(0.0, 1.0)

        t_uv_raw = self._blended_uv_fiber(w_soft, seeds)

        rho0, gamma = 0.5, 0.05
        m = torch.sigmoid((rho - rho0) / gamma).unsqueeze(1)
        t_uv = t_uv_raw * m

        fiber3d = self.map_to_3d(t_uv, Xu=Xu, Xv=Xv)
        h = self.height(h_raw, ref_tensor=points_uv)

        return {
            "w_soft": w_soft,
            "d": d,
            "M": M,
            "seeds": seeds,
            "rho": rho,
            "rho_v": rho_v,
            "rho_b": rho_b,
            "t_uv_raw": t_uv_raw,
            "t_uv": t_uv,
            "fiber3d": fiber3d,
            "h": h,
            "w_geo": w_geo,
            "pair_strength": pair_strength,
            "band_ij": band_ij,
            "pair_relevance": pair_relevance,
            "edge_field": edge_field,
            "gap_thr": gap_thr,
            "big_thr": big_thr,
            "alpha": alpha,
            "eta": eta,
            "boundary_alpha": alpha_b,
            "boundary_width": (
                self.boundary_width(points_uv, boundary_width_raw)
                if self.use_boundary_attachment
                else torch.zeros((), device=points_uv.device, dtype=points_uv.dtype)
            ),
            "boundary_beta": (
                self.boundary_beta(points_uv, boundary_beta_raw)
                if self.use_boundary_attachment
                else torch.zeros((), device=points_uv.device, dtype=points_uv.dtype)
            ),
        }

    def forward(
        self,
        points_uv,
        Xu,
        Xv,
        tau,
        seeds_raw,
        w_raw,
        h_raw=None,
        theta=None,
        a_raw=None,
        points_face_id=None,
        boundary_uv=None,
        boundary_face_id=None,
        gap_thr_raw=None,
        big_thr_raw=None,
        alpha_raw=None,
        eta_raw=None,
        boundary_width_raw=None,
        boundary_alpha_raw=None,
        boundary_beta_raw=None,
        seed_gates=None,
    ):
        return self.evaluate_at_uv(
            points_uv=points_uv,
            Xu=Xu,
            Xv=Xv,
            tau=tau,
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            h_raw=h_raw,
            theta=theta,
            a_raw=a_raw,
            points_face_id=points_face_id,
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
            gap_thr_raw=gap_thr_raw,
            big_thr_raw=big_thr_raw,
            alpha_raw=alpha_raw,
            eta_raw=eta_raw,
            boundary_width_raw=boundary_width_raw,
            boundary_alpha_raw=boundary_alpha_raw,
            boundary_beta_raw=boundary_beta_raw,
            seed_gates=seed_gates,
        )