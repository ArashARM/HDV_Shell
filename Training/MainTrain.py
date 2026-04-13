from dataclasses import dataclass
import math
import os
from datetime import datetime

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from Utils.TimelapseRecorder import TimelapseRecorder
from tqdm.auto import tqdm
import numpy as np
from scipy.interpolate import griddata

import pyvista as pv
try:
    pv.set_jupyter_backend("trame")
except Exception:
    pass

import matplotlib.pyplot as plt


@dataclass
class TrainingConfig:
    seed_number: int = 15
    use_Metric_anisotropy: bool = True
    fixed_height: float | None = None
    target_volfrac: float = 0.5
    seed_repulsion_sigma: float = 0.08
    boundary_margin: float = 0.05
    freeze_w: bool = False
    use_boundary_attachment: bool = True
    boundary_volume_assist: float = 0.10

    # boundary defaults / fallback values used by decoder if PPNet does not predict them
    boundary_attach_width: float = 0.03
    boundary_attach_beta: float = 0.01
    boundary_attach_alpha: float = 0.35

    boundary_attach_width_min: float = 0.005
    boundary_attach_width_max: float = 0.10

    boundary_attach_alpha_min: float = 0.05
    boundary_attach_alpha_max: float = 1.00

    boundary_attach_beta_min: float = 0.003
    boundary_attach_beta_max: float = 0.05

    gate_sharpen_gamma: float = 4.0

    w_min: float = 0.005
    w_max: float = 0.5  # Deprecated: pairwise widths now use 0.8 * seed distance as the upper bound.

    lam_fem: float = 1.0
    lam_vol: float = 2.0
    lam_rep: float = 0.5
    lam_bnd: float = 0.5

    lam_strut: float = 0.02
    lam_strut_edge: float = 1.0
    lam_strut_void: float = 0.25

    comp_normalize_by: float | None = 1e10
    normalize_losses: bool = True
    fem_density_floor: float = 0.02
    skip_bad_fem_steps: bool = True

    num_steps: int = 10000
    context_vector_size: int = 8

    tau: float = 0.02
    beta: float = 0.05

    lr_seed_refine: float = 1e-1
    lr_delta_head: float = 2e-4
    lr_mlp: float = 2e-4
    lr_w_head: float = 2e-4
    lr_h_head: float = 2e-4
    lr_boundary_heads: float = 2e-4

    log_every: int = 50
    early_stop_start: int = 300
    patience: int = 300
    min_delta: float = 1e-4

    use_gating: bool = True
    lr_gate_head: float = 5e-5
    gate_active_threshold: float = 0.5
    gate_eps: float = 1e-8
    gate_bias_init: float = math.log(0.45 / 0.55)
    Gating_binirize_sharpness: float = 20.0
    Gating_binirize_limit: float = 0.45
    use_hard_gate_mask: bool = False
    hard_gate_start_frac: float = 0.90
    hard_gate_allow_early_stability: bool = True
    hard_gate_stability_patience: int = 200
    hard_gate_min_frac_for_stability: float = 0.25

    lam_gate_count: float = 2.0
    gate_target_count: float = 10.0
    lam_gate_binary: float = 0.2
    gate_binary_warmup_steps: int = 40
    gate_warmup_steps: int = 20

    predict_boundary_width: bool = True

    eps: float = 1e-12

    use_boundary_weighted_volume: bool = False
    boundary_vol_weight: float = 0.20
    effective_volume_power: float = 2.0
    lam_vol_effective: float = 0.5
    lam_vol_sharp: float = 0.5
    sharp_vol_start_frac: float = 0.6
    sharp_vol_ramp_frac: float = 0.3

    Offset_scale: float = 1.00
    scheduler_milestones: tuple[float, ...] = (80, 160)
    scheduler_gamma: float = 0.5

    save_fem_debug_history: bool = True
    grad_clip_norm: float | None = 1.0

    tensorboard_enabled: bool = True
    tensorboard_log_root: str = "runs"
    experiment_name: str | None = None
    tb_flush_secs: int = 10
    tb_log_histograms_every: int = 200

    MakeTimelaps: bool = True

    timelapse_frame_step: int = 20
    TM_laps_res_u: int = 100
    TM_laps_res_v: int = 100
    TM_laps_Thr: float = 0.45

class RunningNorm:
    def __init__(self, momentum: float = 0.99, eps: float = 1e-12):
        self.val = None
        self.momentum = momentum
        self.eps = eps

    def update(self, x: float) -> float:
        x = abs(float(x))
        if not math.isfinite(x):
            return max(self.val if self.val is not None else 1.0, 1e-8)

        x = x + self.eps
        if self.val is None:
            self.val = x
        else:
            self.val = self.momentum * self.val + (1.0 - self.momentum) * x
        return max(self.val, 1e-8)


class NN_Trainer:
    def __init__(
        self,
        generator,
        viz,
        decoder_cls,
        ppnet_cls,
        fem,
        shell_problem,
        config: TrainingConfig,
    ):
        self.generator = generator
        self.viz = viz
        self.decoder_cls = decoder_cls
        self.ppnet_cls = ppnet_cls
        self.fem = fem
        self.shell_problem = shell_problem
        self.cfg = config

        self.last_fem_debug = {}
        self.fem_debug_history = []

        self.writer = None
        self.tensorboard_log_dir = None
        self._init_tensorboard()

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------

    def _init_tensorboard(self):
        if not self.cfg.tensorboard_enabled:
            return

        exp_name = self.cfg.experiment_name
        if exp_name is None or str(exp_name).strip() == "":
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = os.path.join(self.cfg.tensorboard_log_root, exp_name)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=self.cfg.tb_flush_secs,
        )
        self.tensorboard_log_dir = log_dir

        cfg_lines = [f"{k}: {v}" for k, v in vars(self.cfg).items()]
        self.writer.add_text("config", "\n".join(cfg_lines), global_step=0)

        print(f"TensorBoard log dir: {self.tensorboard_log_dir}")

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _true_open_boundary_idx(self, ft, tol=None):
        if ("boundary_idx_ring1" not in ft) or ft["boundary_idx_ring1"] is None:
            return torch.empty(0, dtype=torch.long, device=ft["uv"].device)

        bidx = torch.unique(ft["boundary_idx_ring1"].to(dtype=torch.long))
        if bidx.numel() == 0:
            return bidx

        uv = ft["uv"]
        u = uv[:, 0]
        v = uv[:, 1]

        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        if tol is None:
            u_span = (u.max() - u.min()).abs()
            v_span = (v.max() - v.min()).abs()
            base_span = torch.maximum(
                u_span,
                v_span,
            ).clamp_min(torch.as_tensor(1.0, device=uv.device, dtype=uv.dtype))
            tol = 1e-4 * float(base_span.detach().item())

        ub = u[bidx]
        vb = v[bidx]
        keep = torch.ones_like(bidx, dtype=torch.bool)

        if u_periodic:
            umin = u.min()
            umax = u.max()
            is_u_seam = (ub - umin).abs() <= tol
            is_u_seam = is_u_seam | ((ub - umax).abs() <= tol)
            keep = keep & (~is_u_seam)

        if v_periodic:
            vmin = v.min()
            vmax = v.max()
            is_v_seam = (vb - vmin).abs() <= tol
            is_v_seam = is_v_seam | ((vb - vmax).abs() <= tol)
            keep = keep & (~is_v_seam)

        return bidx[keep]

    @staticmethod
    def _to_float_if_finite(x):
        if isinstance(x, torch.Tensor):
            x = x.reshape(())
            if torch.isfinite(x).item():
                return float(x.detach().item())
            return None
        try:
            x = float(x)
            return x if math.isfinite(x) else None
        except Exception:
            return None

    def _tb_add_scalar(self, tag: str, value, step: int):
        if self.writer is None:
            return
        v = self._to_float_if_finite(value)
        if v is not None:
            self.writer.add_scalar(tag, v, step)

    def _tb_add_histogram(self, tag: str, value: torch.Tensor, step: int):
        if self.writer is None or value is None:
            return
        try:
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                finite_mask = torch.isfinite(value)
                if finite_mask.any():
                    self.writer.add_histogram(tag, value[finite_mask].detach().cpu(), step)
        except Exception:
            pass

    def _tb_log_step(
        self,
        step: int,
        row: dict,
        rho: torch.Tensor,
        rho_boundary: torch.Tensor,
        rho_v_all: torch.Tensor,
        fiber_surface: torch.Tensor,
        seeds_list: list[torch.Tensor],
        pred_list: list[dict],
    ):
        if self.writer is None:
            return

        self._tb_add_scalar("Loss/Total", row["L_total"], step)
        self._tb_add_scalar("Loss/Volume", row["loss_vol"], step)
        self._tb_add_scalar("Loss/Repulsion", row["loss_rep"], step)
        self._tb_add_scalar("Loss/Boundary", row["loss_bnd"], step)
        self._tb_add_scalar("Loss/Strut", row["loss_strut"], step)
        self._tb_add_scalar("Loss/StrutEdge", row["loss_strut_edge"], step)
        self._tb_add_scalar("Loss/StrutVoid", row["loss_strut_void"], step)
        self._tb_add_scalar("Loss/FEM", row["loss_fem"], step)
        self._tb_add_scalar("Loss/Compliance", row["loss_comp"], step)

        self._tb_add_scalar("Physics/ComplianceRaw", row["comp"], step)
        self._tb_add_scalar("Physics/VolumeFraction", row["vol_frac"], step)
        self._tb_add_scalar("Physics/VolumeFractionEffective", row["vol_frac_eff"], step)
        self._tb_add_scalar("Physics/VolumeFractionSharp", row["vol_frac_sharp"], step)
        self._tb_add_scalar("Physics/VolumeDeviation", row["vol_dev"], step)
        self._tb_add_scalar("Physics/VolumeDeviationEffective", row["vol_dev_eff"], step)
        self._tb_add_scalar("Loss/VolumeSharp", row["loss_vol_sharp"], step)
        self._tb_add_scalar("Train/SharpVolRamp", row["sharp_vol_ramp"], step)
        self._tb_add_scalar("Physics/RhoBoundaryMean", row["rho_boundary_mean"], step)
        self._tb_add_scalar("Physics/RhoVoronoiMean", row["rho_v_mean"], step)
        self._tb_add_scalar("Physics/WGeoMean", row["w_geo_mean"], step)

        self._tb_add_scalar("Density/Min", row["rho_min"], step)
        self._tb_add_scalar("Density/Mean", row["rho_mean"], step)
        self._tb_add_scalar("Density/Max", row["rho_max"], step)

        self._tb_add_scalar("Train/DeltaRho", row["drho"], step)
        self._tb_add_scalar("Train/DeltaSeed", row["dseed"], step)
        self._tb_add_scalar("Train/GradMean", row["grad_mean"], step)
        self._tb_add_scalar("Train/BestScore", row["best_score"], step)
        self._tb_add_scalar("Train/BestStep", row["best_step"], step)
        self._tb_add_scalar("Train/FEMValid", 1.0 if row["fem_valid"] else 0.0, step)
        self._tb_add_scalar(
            "Train/OptimizerStepSkipped",
            1.0 if row["optimizer_step_skipped"] else 0.0,
            step,
        )
        self._tb_add_scalar("Geometry/HMean", row["h_mean"], step)

        self._tb_add_scalar("Boundary/Width", row["boundary_width_mean"], step)
        self._tb_add_scalar("Boundary/Alpha", row["boundary_alpha_mean"], step)
        self._tb_add_scalar("Boundary/Beta", row["boundary_beta_mean"], step)

        self._tb_add_scalar("Metric/ThetaMean", row["theta_mean"], step)
        self._tb_add_scalar("Metric/AMean", row["a_metric_mean"], step)

        self._tb_add_scalar("gate/active_count_total", row["active_count_total"], step)
        self._tb_add_scalar("gate/active_count_mean", row["active_count_mean"], step)
        self._tb_add_scalar("gate/active_frac_mean", row["active_frac_mean"], step)
        self._tb_add_scalar("gate/raw/min", row["gate_raw_min"], step)
        self._tb_add_scalar("gate/raw/mean", row["gate_raw_mean"], step)
        self._tb_add_scalar("gate/raw/max", row["gate_raw_max"], step)
        self._tb_add_scalar("gate/sharp/min", row["gate_sharp_min"], step)
        self._tb_add_scalar("gate/sharp/mean", row["gate_sharp_mean"], step)
        self._tb_add_scalar("gate/sharp/max", row["gate_sharp_max"], step)
        self._tb_add_scalar("gate/decoder/min", row["gate_decoder_min"], step)
        self._tb_add_scalar("gate/decoder/mean", row["gate_decoder_mean"], step)
        self._tb_add_scalar("gate/decoder/max", row["gate_decoder_max"], step)
        self._tb_add_scalar("gate/hard_mask_on", row["hard_gate_mask_on"], step)
        self._tb_add_scalar("Loss/Gate", row["loss_gate"], step)
        self._tb_add_scalar("gate/lam_eff", row["lam_gate_eff"], step)
        self._tb_add_scalar("Loss/GateBinary", row["loss_gate_binary"], step)
        self._tb_add_scalar("gate/lam_binary_eff", row["lam_gate_binary_eff"], step)

        fiber_norm = torch.linalg.norm(fiber_surface, dim=1)
        if fiber_norm.numel() > 0:
            self._tb_add_scalar("Fiber/NormMean", fiber_norm.mean(), step)
            self._tb_add_scalar("Fiber/NormMin", fiber_norm.min(), step)
            self._tb_add_scalar("Fiber/NormMax", fiber_norm.max(), step)

        if step % self.cfg.tb_log_histograms_every == 0 or step == self.cfg.num_steps - 1:
            self._tb_add_histogram("Density/Rho", rho, step)
            self._tb_add_histogram("Density/RhoBoundary", rho_boundary, step)
            self._tb_add_histogram("Density/RhoVoronoi", rho_v_all, step)
            self._tb_add_histogram("Fiber/Norm", fiber_norm, step)

            if len(seeds_list) > 0:
                all_seeds = torch.cat(seeds_list, dim=0)
                self._tb_add_histogram("Seeds/All", all_seeds, step)
                if all_seeds.shape[1] >= 1:
                    self._tb_add_histogram("Seeds/U", all_seeds[:, 0], step)
                if all_seeds.shape[1] >= 2:
                    self._tb_add_histogram("Seeds/V", all_seeds[:, 1], step)

            w_geo_vals = []
            for p in pred_list:
                if "w_geo" in p and p["w_geo"] is not None:
                    w_geo_vals.append(self._pair_upper_values(p["w_geo"]))
            if len(w_geo_vals) > 0:
                self._tb_add_histogram("Geometry/WGeo", torch.cat(w_geo_vals, dim=0), step)
            
            bw_vals = []
            ba_vals = []
            bb_vals = []
            h_vals = []
            theta_vals = []
            a_vals = []

            for p in pred_list:
                if "boundary_width" in p and p["boundary_width"] is not None:
                    bw_vals.append(p["boundary_width"].reshape(-1))
                if "boundary_alpha" in p and p["boundary_alpha"] is not None:
                    ba_vals.append(p["boundary_alpha"].reshape(-1))
                if "boundary_beta" in p and p["boundary_beta"] is not None:
                    bb_vals.append(p["boundary_beta"].reshape(-1))
                if "h" in p and p["h"] is not None:
                    h_vals.append(p["h"].reshape(-1))
                if "theta" in p and p["theta"] is not None:
                    theta_vals.append(p["theta"].reshape(-1))
                if "a_metric" in p and p["a_metric"] is not None:
                    a_vals.append(p["a_metric"].reshape(-1))

            if bw_vals: self._tb_add_histogram("Boundary/WidthHist", torch.cat(bw_vals, dim=0), step)
            if ba_vals: self._tb_add_histogram("Boundary/AlphaHist", torch.cat(ba_vals, dim=0), step)
            if bb_vals: self._tb_add_histogram("Boundary/BetaHist", torch.cat(bb_vals, dim=0), step)
            if h_vals: self._tb_add_histogram("Geometry/HHist", torch.cat(h_vals, dim=0), step)
            if theta_vals: self._tb_add_histogram("Metric/ThetaHist", torch.cat(theta_vals, dim=0), step)
            if a_vals: self._tb_add_histogram("Metric/AHist", torch.cat(a_vals, dim=0), step)

            gate_logit_vals = []
            gate_raw_vals = []
            gate_sharp_vals = []
            gate_decoder_vals = []
            for p in pred_list:
                if p.get("gate_logits") is not None:
                    gate_logit_vals.append(p["gate_logits"].reshape(-1))
                if p.get("gate_probs_raw") is not None:
                    gate_raw_vals.append(p["gate_probs_raw"].reshape(-1))
                if p.get("gate_probs_sharp") is not None:
                    gate_sharp_vals.append(p["gate_probs_sharp"].reshape(-1))
                if p.get("gate_probs_decoder") is not None:
                    gate_decoder_vals.append(p["gate_probs_decoder"].reshape(-1))

            if gate_logit_vals:
                self._tb_add_histogram("gate/logits", torch.cat(gate_logit_vals, dim=0), step)
            if gate_raw_vals:
                self._tb_add_histogram("gate/raw", torch.cat(gate_raw_vals, dim=0), step)
            if gate_sharp_vals:
                self._tb_add_histogram("gate/sharp", torch.cat(gate_sharp_vals, dim=0), step)
            if gate_decoder_vals:
                self._tb_add_histogram("gate/decoder", torch.cat(gate_decoder_vals, dim=0), step)

        if self.last_fem_debug:
            dbg = self.last_fem_debug
            for key in [
                "density_raw_min",
                "density_raw_mean",
                "density_raw_max",
                "density_min",
                "density_mean",
                "density_max",
                "fiber_norm_min",
                "fiber_norm_mean",
                "fiber_norm_max",
                "void_fraction_lt_1e_2_raw",
                "void_fraction_lt_5e_2_raw",
                "void_fraction_lt_floor_raw",
            ]:
                if key in dbg:
                    self._tb_add_scalar(f"FEMDebug/{key}", dbg[key], step)

            if "fem_valid" in dbg:
                self._tb_add_scalar("FEMDebug/Valid", 1.0 if dbg["fem_valid"] else 0.0, step)

            if dbg.get("failure_reason"):
                self.writer.add_text("FEMDebug/FailureReason", str(dbg["failure_reason"]), step)

    # ------------------------------------------------------------------
    # Losses / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def volume_loss_constant_height(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        vol_frac = (rho * A_v).sum() / (A_v.sum() + eps)
        vol_loss = (vol_frac - target_volfrac) ** 2
        return vol_loss

    @staticmethod
    def powered_volume_fraction(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        rho_eff = rho.clamp(0.0, 1.0).pow(power)
        return (rho_eff * A_v).sum() / (A_v.sum() + eps)

    @classmethod
    def volume_loss_powered(
        cls,
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vol_frac_eff = cls.powered_volume_fraction(rho=rho, A_v=A_v, power=power, eps=eps)
        loss = (vol_frac_eff - target_volfrac) ** 2
        return loss, vol_frac_eff

    @staticmethod
    def ramp_weight(step: int, total_steps: int, start_frac: float, ramp_frac: float) -> float:
        if total_steps <= 0:
            return 0.0
        start_step = max(int(start_frac * total_steps), 0)
        ramp_steps = max(int(ramp_frac * total_steps), 1)
        if step <= start_step:
            return 0.0
        if step >= start_step + ramp_steps:
            return 1.0
        return float(step - start_step) / float(ramp_steps)

    @staticmethod
    def gate_target_loss(
        gates: torch.Tensor | None,
        target_mean: float,
    ):
        if gates is None:
            return None
        target = torch.as_tensor(target_mean, device=gates.device, dtype=gates.dtype)
        return (gates.mean() - target) ** 2
    
    @staticmethod
    def gate_count_loss(gates: torch.Tensor | None, target_count: float):
        if gates is None:
            return None
        target = torch.as_tensor(target_count, device=gates.device, dtype=gates.dtype)
        return (gates.sum() - target) ** 2
    @staticmethod
    def gate_binary_loss(gates: torch.Tensor | None):
        if gates is None:
            return None
        one = torch.ones((), device=gates.device, dtype=gates.dtype)
        return (gates * (one - gates)).mean()
    @staticmethod
    def volume_loss_with_boundary_discount(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        rho_boundary: torch.Tensor,
        target_volfrac: float,
        boundary_weight: float = 0.20,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = 1.0 - rho_boundary + boundary_weight * rho_boundary
        vol_frac_eff = (rho * weight * A_v).sum() / ((weight * A_v).sum() + eps)
        loss = (vol_frac_eff - target_volfrac) ** 2
        return loss, vol_frac_eff

    @staticmethod
    def seed_repulsion_term(
        seeds: torch.Tensor,
        gates: torch.Tensor | None = None,
        sigma: float = 0.08,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        S = seeds.shape[0]
        d2 = torch.cdist(seeds, seeds).pow(2)

        eye = torch.eye(S, dtype=torch.bool, device=seeds.device)
        mask = ~eye
        K = torch.exp(-d2 / (sigma**2 + eps))

        if gates is None:
            return K[mask].mean()

        g = gates.view(-1)
        G = g[:, None] * g[None, :]
        return (G * K)[mask].sum() / (G[mask].sum() + eps)

    @staticmethod
    def boundary_repulsion_term(
        seeds: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        gates: torch.Tensor | None = None,
        margin: float = 0.05,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if boundary_uv is None or boundary_uv.numel() == 0:
            return torch.zeros((), dtype=seeds.dtype, device=seeds.device)

        dmin = torch.cdist(seeds, boundary_uv).amin(dim=1)
        penalty = torch.exp(-dmin / (margin + eps))

        if gates is None:
            return penalty.mean()

        g = gates.view(-1)
        return (g * penalty).sum() / (g.sum() + eps)

    @staticmethod
    def compliance_loss(
        comp: torch.Tensor,
        normalize_by: float | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        comp = comp.reshape(())
        if normalize_by is not None:
            return comp / (float(normalize_by) + eps)
        return comp

    @staticmethod
    def strutness_loss_from_edge_field(
        rho: torch.Tensor,
        edge_field: torch.Tensor,
        lam_edge: float = 1.0,
        lam_void: float = 1.0,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_field = edge_field.clamp(0.0, 1.0)
        void_field = 1.0 - edge_field

        loss_edge = (edge_field * (1.0 - rho).pow(2)).sum() / (edge_field.sum() + eps)
        loss_void = (void_field * rho.pow(2)).sum() / (void_field.sum() + eps)

        loss = lam_edge * loss_edge + lam_void * loss_void
        return loss, loss_edge, loss_void
    


    @staticmethod
    def _scalar_tensor_is_finite(x: torch.Tensor) -> bool:
        return bool(torch.isfinite(x).reshape(()).detach().item())

    @staticmethod
    def _require_decoder_keys(decoder_out: dict, required_keys: list[str]):
        missing = [k for k in required_keys if k not in decoder_out]
        if missing:
            raise ValueError(
                f"Decoder output missing required keys: {missing}. "
                f"Available keys: {list(decoder_out.keys())}"
            )

    def _record_invalid_fem_debug(
        self,
        debug: dict,
        reason: str,
        save_debug_history: bool,
    ):
        debug = dict(debug)
        debug["fem_valid"] = False
        debug["failure_reason"] = reason
        self.last_fem_debug = debug
        if save_debug_history:
            self.fem_debug_history.append(debug.copy())

    
    def fem_loss(
        self,
        rho_surface: torch.Tensor,
        fiber_surface: torch.Tensor,
        comp_normalize_by: float | None = None,
        density_floor: float = 0.02,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> dict:
        device = rho_surface.device
        dtype = rho_surface.dtype

        fem_fields = self.shell_problem.build_fem_fields_from_decoder_torch(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
        )
        density_raw = fem_fields["density"].to(device=device, dtype=dtype)
        density = density_raw.clamp_min(density_floor)

        phi = fem_fields["phi"].to(device=device, dtype=dtype)
        theta = fem_fields["theta"].to(device=device, dtype=dtype)

        fiber_norm = torch.linalg.norm(fiber_surface, dim=1)

        debug = {
            "rho_surface_shape": tuple(rho_surface.shape),
            "fiber_surface_shape": tuple(fiber_surface.shape),
            "density_shape": tuple(density.shape),
            "phi_shape": tuple(phi.shape),
            "theta_shape": tuple(theta.shape),
            "density_floor": float(density_floor),
            "density_raw_min": float(density_raw.min().detach().item()),
            "density_raw_mean": float(density_raw.mean().detach().item()),
            "density_raw_max": float(density_raw.max().detach().item()),
            "density_min": float(density.min().detach().item()),
            "density_mean": float(density.mean().detach().item()),
            "density_max": float(density.max().detach().item()),
            "phi_has_nan": bool(torch.isnan(phi).any().detach().item()),
            "phi_has_inf": bool(torch.isinf(phi).any().detach().item()),
            "theta_has_nan": bool(torch.isnan(theta).any().detach().item()),
            "theta_has_inf": bool(torch.isinf(theta).any().detach().item()),
            "fiber_has_nan": bool(torch.isnan(fiber_surface).any().detach().item()),
            "fiber_has_inf": bool(torch.isinf(fiber_surface).any().detach().item()),
            "fiber_norm_min": float(fiber_norm.min().detach().item()),
            "fiber_norm_mean": float(fiber_norm.mean().detach().item()),
            "fiber_norm_max": float(fiber_norm.max().detach().item()),
            "void_fraction_lt_1e_2_raw": float((density_raw < 1e-2).float().mean().detach().item()),
            "void_fraction_lt_5e_2_raw": float((density_raw < 5e-2).float().mean().detach().item()),
            "void_fraction_lt_floor_raw": float((density_raw < density_floor).float().mean().detach().item()),
        }

        if (
            debug["phi_has_nan"] or debug["phi_has_inf"] or
            debug["theta_has_nan"] or debug["theta_has_inf"] or
            debug["fiber_has_nan"] or debug["fiber_has_inf"]
        ):
            reason = "Invalid phi/theta/fiber fields before FEM solve"
            self._record_invalid_fem_debug(debug, reason, save_debug_history)
            nan_scalar = torch.full((), float("nan"), dtype=dtype, device=device)
            return {
                "fem_total": nan_scalar,
                "comp": nan_scalar,
                "compliance_loss": nan_scalar,
                "fem_valid": False,
                "failure_reason": reason,
            }

        try:
            _stress_unused, comp = self.fem(density, phi, theta, penal=3)
        except Exception as e:
            reason = f"FEM solve raised exception: {repr(e)}"
            self._record_invalid_fem_debug(debug, reason, save_debug_history)
            nan_scalar = torch.full((), float("nan"), dtype=dtype, device=device)
            return {
                "fem_total": nan_scalar,
                "comp": nan_scalar,
                "compliance_loss": nan_scalar,
                "fem_valid": False,
                "failure_reason": reason,
            }

        debug.update({
            "comp_is_finite": self._scalar_tensor_is_finite(comp),
            "comp_value": float(torch.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
        })

        if not debug["comp_is_finite"]:
            reason = "Non-finite compliance returned by FEM solve"
            self._record_invalid_fem_debug(debug, reason, save_debug_history)
            nan_scalar = torch.full((), float("nan"), dtype=dtype, device=device)
            return {
                "fem_total": nan_scalar,
                "comp": comp,
                "compliance_loss": nan_scalar,
                "fem_valid": False,
                "failure_reason": reason,
            }

        loss_comp = self.compliance_loss(
            comp=comp,
            normalize_by=comp_normalize_by,
            eps=eps,
        )

        debug.update({
            "loss_comp_is_finite": self._scalar_tensor_is_finite(loss_comp),
            "fem_total_is_finite": self._scalar_tensor_is_finite(loss_comp),
            "loss_comp_value": float(torch.nan_to_num(loss_comp, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
            "fem_total_value": float(torch.nan_to_num(loss_comp, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
        })

        fem_valid = debug["loss_comp_is_finite"]
        debug["fem_valid"] = fem_valid
        debug["failure_reason"] = None if fem_valid else "Non-finite compliance loss"

        self.last_fem_debug = debug
        if save_debug_history:
            self.fem_debug_history.append(debug.copy())

        return {
            "fem_total": loss_comp,
            "comp": comp,
            "compliance_loss": loss_comp,
            "fem_valid": fem_valid,
            "failure_reason": debug["failure_reason"],
        }

    def total_loss(
        self,
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        seeds: torch.Tensor,
        boundary_uv: torch.Tensor | None = None,
        fiber_surface: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
        w_vol: float = 1.0,
        w_seed: float = 1.0,
        w_boundary: float = 1.0,
        w_strut: float = 0.0,
        w_fem: float = 0.0,
        comp_normalize_by: float | None = None,
        density_floor: float = 0.02,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> dict:
        sigma = self.cfg.seed_repulsion_sigma
        margin = self.cfg.boundary_margin

        loss_vol = self.volume_loss_constant_height(
            rho=rho,
            A_v=A_v,
            target_volfrac=target_volfrac,
            eps=eps,
        )

        loss_seed = self.seed_repulsion_term(
            seeds=seeds,
            gates=gates,
            sigma=sigma,
            eps=eps,
        )

        loss_boundary = self.boundary_repulsion_term(
            seeds=seeds,
            boundary_uv=boundary_uv,
            gates=gates,
            margin=margin,
            eps=eps,
        )

        loss_strut = torch.zeros((), dtype=rho.dtype, device=rho.device)
        loss_strut_edge = torch.zeros((), dtype=rho.dtype, device=rho.device)
        loss_strut_void = torch.zeros((), dtype=rho.dtype, device=rho.device)

        total = (
            w_vol * loss_vol +
            w_seed * loss_seed +
            w_boundary * loss_boundary +
            w_strut * loss_strut
        )

        fem_out = {
            "fem_total": torch.zeros((), dtype=rho.dtype, device=rho.device),
            "comp": torch.zeros((), dtype=rho.dtype, device=rho.device),
            "compliance_loss": torch.zeros((), dtype=rho.dtype, device=rho.device),
            "fem_valid": True,
            "failure_reason": None,
        }

        if w_fem != 0.0:
            if fiber_surface is None:
                raise ValueError("fiber_surface must be provided when w_fem != 0")

            fem_out = self.fem_loss(
                rho_surface=rho,
                fiber_surface=fiber_surface,
                comp_normalize_by=comp_normalize_by,
                density_floor=density_floor,
                eps=eps,
                save_debug_history=save_debug_history,
            )

            if fem_out["fem_valid"]:
                total = total + w_fem * fem_out["fem_total"]

        return {
            "total": total,
            "volume": loss_vol,
            "seed_repulsion": loss_seed,
            "boundary_repulsion": loss_boundary,
            "strut": loss_strut,
            "strut_edge": loss_strut_edge,
            "strut_void": loss_strut_void,
            "fem_total": fem_out["fem_total"],
            "comp": fem_out["comp"],
            "compliance_loss": fem_out["compliance_loss"],
            "fem_valid": fem_out["fem_valid"],
            "fem_failure_reason": fem_out["failure_reason"],
        }

    # ------------------------------------------------------------------
    # Model / optimizer builders
        # ------------------------------------------------------------------
    def _build_single_face_models(
        self,
        device,
        seed_number,
        boundary_idx_ring1,
        u_periodic,
        v_periodic,
        use_Metric_anisotropy,
        context_vector_size,
        freeze_w,
    ):
        face_u_periodic = torch.tensor([bool(u_periodic)], dtype=torch.bool, device=device)
        face_v_periodic = torch.tensor([bool(v_periodic)], dtype=torch.bool, device=device)
        seed_face_id = torch.zeros(seed_number, dtype=torch.long, device=device)

        decoder = self.decoder_cls(
            n_seeds=seed_number,
            boundary_solid_idx=boundary_idx_ring1,
            seed_face_id=seed_face_id,
            face_u_periodic=face_u_periodic,
            face_v_periodic=face_v_periodic,
            use_Metric_anisotropy=use_Metric_anisotropy,
            w_min=self.cfg.w_min,
            fixed_height=self.cfg.fixed_height,

            use_boundary_attachment=self.cfg.use_boundary_attachment,
            boundary_attach_width=self.cfg.boundary_attach_width,
            boundary_attach_beta=self.cfg.boundary_attach_beta,
            boundary_attach_alpha=self.cfg.boundary_attach_alpha,

            boundary_attach_width_min=self.cfg.boundary_attach_width_min,
            boundary_attach_width_max=self.cfg.boundary_attach_width_max,
            boundary_attach_alpha_min=self.cfg.boundary_attach_alpha_min,
            boundary_attach_alpha_max=self.cfg.boundary_attach_alpha_max,
            boundary_attach_beta_min=self.cfg.boundary_attach_beta_min,
            boundary_attach_beta_max=self.cfg.boundary_attach_beta_max,
        ).to(device)

        ppnet = self.ppnet_cls(
            context_dim=context_vector_size,
            n_seeds=seed_number,
            use_Metric_anisotropy=use_Metric_anisotropy,
            predict_height=(self.cfg.fixed_height is None),
            use_gating=self.cfg.use_gating,
            predict_boundary_width=self.cfg.predict_boundary_width,
            freeze_w=freeze_w,
            gate_bias_init=self.cfg.gate_bias_init,
        ).to(device)

        return decoder, ppnet

    def _build_face_models(self, face_tensors, device):
        cfg = self.cfg
        decoders = []
        ppnets = []

        for ft in face_tensors:
            decoder, ppnet = self._build_single_face_models(
                device=device,
                seed_number=cfg.seed_number,
                boundary_idx_ring1=ft["boundary_idx_ring1"],
                u_periodic=ft.get("u_periodic", False),
                v_periodic=ft.get("v_periodic", False),
                use_Metric_anisotropy=cfg.use_Metric_anisotropy,
                context_vector_size=cfg.context_vector_size,
                freeze_w=cfg.freeze_w,
            )
            decoders.append(decoder)
            ppnets.append(ppnet)

        return decoders, ppnets

    def _build_optimizer(self, ppnets, decoders):
        cfg = self.cfg
        param_groups = []

        for ppnet in ppnets:
            param_groups.extend([
                {"params": ppnet.seed_refine.parameters(), "lr": cfg.lr_seed_refine},
                {"params": ppnet.delta_head.parameters(), "lr": cfg.lr_delta_head},
                {"params": ppnet.mlp.parameters(), "lr": cfg.lr_mlp},
            ])

            if hasattr(ppnet, "w_head"):
                param_groups.append({"params": ppnet.w_head.parameters(), "lr": cfg.lr_w_head})

            if (self.cfg.fixed_height is None) and hasattr(ppnet, "h_head"):
                param_groups.append({"params": ppnet.h_head.parameters(), "lr": cfg.lr_h_head})

            if hasattr(ppnet, "gate_head"):
                param_groups.append({"params": ppnet.gate_head.parameters(), "lr": cfg.lr_gate_head})

            if cfg.use_Metric_anisotropy:
                if hasattr(ppnet, "theta_head"):
                    param_groups.append({"params": ppnet.theta_head.parameters(), "lr": cfg.lr_mlp})
                if hasattr(ppnet, "a_head"):
                    param_groups.append({"params": ppnet.a_head.parameters(), "lr": cfg.lr_mlp})

            if cfg.predict_boundary_width and hasattr(ppnet, "boundary_width_head"):
                param_groups.append({
                    "params": ppnet.boundary_width_head.parameters(),
                    "lr": cfg.lr_boundary_heads,
                })

        return torch.optim.Adam(param_groups)

    @staticmethod
    def _pair_upper_values(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError("Expected tensor for pair reduction")
        if t.ndim < 2:
            return t.reshape(-1)

        mask = torch.triu(
            torch.ones(t.shape[-2], t.shape[-1], device=t.device, dtype=torch.bool),
            diagonal=1,
        )
        vals = t[..., mask]
        if vals.numel() == 0:
            return t.reshape(-1)
        return vals.reshape(-1)
    
    def _collect_decoder_param_logs(self, decoder, pred_i: dict, ref_tensor: torch.Tensor) -> dict:
        out = {}

        if "w_raw" in pred_i and pred_i["w_raw"] is not None:
            w_geo = decoder.width(pred_i["w_raw"], seeds=pred_i["seeds_raw"])
            out["w_geo"] = self._pair_upper_values(w_geo).mean().reshape(())

        if "h_raw" in pred_i and pred_i["h_raw"] is not None:
            out["h"] = decoder.height(pred_i["h_raw"], ref_tensor=ref_tensor).reshape(())

        if "boundary_width_raw" in pred_i and pred_i["boundary_width_raw"] is not None:
            out["boundary_width"] = decoder.boundary_width(
                ref_tensor, pred_i["boundary_width_raw"]
            ).reshape(())

        if "boundary_alpha_raw" in pred_i and pred_i["boundary_alpha_raw"] is not None:
            out["boundary_alpha"] = decoder.boundary_alpha(
                ref_tensor, pred_i["boundary_alpha_raw"]
            ).reshape(())

        if "boundary_beta_raw" in pred_i and pred_i["boundary_beta_raw"] is not None:
            out["boundary_beta"] = decoder.boundary_beta(
                ref_tensor, pred_i["boundary_beta_raw"]
            ).reshape(())

        if "a_raw" in pred_i and pred_i["a_raw"] is not None:
            a = 0.5 * (2.0 - 0.5) * torch.tanh(pred_i["a_raw"]) + 0.5 * (2.0 + 0.5)
            out["a_metric"] = a.mean().reshape(())

        if "theta" in pred_i and pred_i["theta"] is not None:
            out["theta_mean"] = pred_i["theta"].mean().reshape(())

        return out
    
    
    def _init_face_seeds(self, face_tensors):
        cfg = self.cfg
        uv_init_list = []

        for ft in face_tensors:
            boundary = self._true_open_boundary_idx(ft)
            seed_idx = self.generator.fps_3d(
                ft["points_xyz"],
                cfg.seed_number,
                exclude_idx=boundary,
            )
            uv_init_list.append(ft["uv"][seed_idx].clone())

        return uv_init_list

    def _seed_points_xyz_all_faces(self, seeds_list, face_tensors):
        xyz_parts = []
        for seeds, ft in zip(seeds_list, face_tensors):
            xyz_i = self.generator.seeds_uv_to_xyz_nearest(
                seeds,
                ft["uv"],
                ft["points_xyz"],
            )
            xyz_parts.append(xyz_i)

        if len(xyz_parts) == 0:
            return None

        import numpy as np
        return np.concatenate(xyz_parts, axis=0)

    def _finite_or_default(self, x: torch.Tensor, default: float = float("nan")) -> float:
        if self._scalar_tensor_is_finite(x):
            return float(x.detach().item())
        return default

    def _print_fem_failure(self, step: int):
        print(f"\n=== FEM FAILURE AT STEP {step} ===")
        for k, v in self.last_fem_debug.items():
            print(f"{k}: {v}")
        print("Skipping FEM term for this step.\n")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _validate_face_tensors(self, face_tensors):
        required_keys = [
            "face_id",
            "uv",
            "Xu",
            "Xv",
            "points_xyz",
            "faces_ijk",
            "face_areas",
            "global_vertex_idx",
        ]

        if not isinstance(face_tensors, (list, tuple)) or len(face_tensors) == 0:
            raise ValueError("face_tensors must be a non-empty list.")

        ref_uv = face_tensors[0]["uv"]
        ref_device = ref_uv.device
        ref_dtype = ref_uv.dtype

        for i, ft in enumerate(face_tensors):
            missing = [k for k in required_keys if k not in ft]
            if missing:
                raise ValueError(f"face_tensors[{i}] is missing required keys: {missing}")

            uv = ft["uv"]
            Xu = ft["Xu"]
            Xv = ft["Xv"]
            points_xyz = ft["points_xyz"]
            faces_ijk = ft["faces_ijk"]
            face_areas = ft["face_areas"]
            gidx = ft["global_vertex_idx"]

            if uv.device != ref_device:
                raise ValueError(f"face_tensors[{i}]['uv'] device mismatch: {uv.device} != {ref_device}")
            if uv.dtype != ref_dtype:
                raise ValueError(f"face_tensors[{i}]['uv'] dtype mismatch: {uv.dtype} != {ref_dtype}")

            n_local = uv.shape[0]
            if Xu.shape[0] != n_local or Xv.shape[0] != n_local or points_xyz.shape[0] != n_local:
                raise ValueError(f"face_tensors[{i}] local tensor lengths do not match uv.shape[0]={n_local}")

            if gidx.shape[0] != n_local:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] length mismatch with local vertex count")

            if gidx.dtype != torch.long:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] must be torch.long")

            if gidx.numel() > 0 and int(gidx.min().item()) < 0:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] contains negative indices")

            if faces_ijk.numel() > 0:
                if faces_ijk.dtype != torch.long:
                    raise ValueError(f"face_tensors[{i}]['faces_ijk'] must be torch.long")
                fmin = int(faces_ijk.min().item())
                fmax = int(faces_ijk.max().item())
                if fmin < 0 or fmax >= n_local:
                    raise ValueError(
                        f"face_tensors[{i}]['faces_ijk'] contains invalid local indices "
                        f"(min={fmin}, max={fmax}, n_local={n_local})"
                    )

            if face_areas.ndim != 1:
                raise ValueError(f"face_tensors[{i}]['face_areas'] must be 1D")

            if face_areas.shape[0] != faces_ijk.shape[0]:
                raise ValueError(
                    f"face_tensors[{i}]['face_areas'] length must match number of faces "
                    f"({face_areas.shape[0]} != {faces_ijk.shape[0]})"
                )

    def _safe_weighted_mean(self, values, weights, dtype, device, eps):
        if len(values) == 0:
            return torch.zeros((), dtype=dtype, device=device)
        v = torch.stack(values)
        w = torch.stack(weights).to(dtype=v.dtype, device=v.device)
        return (v * w).sum() / w.sum().clamp_min(eps)

    @staticmethod
    def _build_face_uv_grid(ft, grid_res_u, grid_res_v):
        uv_face = ft["uv"]
        device = uv_face.device
        dtype = uv_face.dtype
        u = uv_face[:, 0]
        v = uv_face[:, 1]

        if bool(ft.get("u_periodic", False)):
            u_lin = torch.linspace(0.0, 1.0, grid_res_u + 1, device=device, dtype=dtype)[:-1]
        else:
            u_lin = torch.linspace(u.min(), u.max(), grid_res_u, device=device, dtype=dtype)

        if bool(ft.get("v_periodic", False)):
            v_lin = torch.linspace(0.0, 1.0, grid_res_v + 1, device=device, dtype=dtype)[:-1]
        else:
            v_lin = torch.linspace(v.min(), v.max(), grid_res_v, device=device, dtype=dtype)

        UU, VV = torch.meshgrid(u_lin, v_lin, indexing="ij")
        uv_grid = torch.stack([UU.reshape(-1), VV.reshape(-1)], dim=1)
        return uv_grid, u_lin, v_lin

    @staticmethod
    def _periodic_uv_min_dist(uv_query, uv_face, u_periodic=False, v_periodic=False, chunk_size=4096):
        if uv_query.numel() == 0 or uv_face.numel() == 0:
            return torch.empty((uv_query.shape[0],), device=uv_query.device, dtype=uv_query.dtype)

        mins = []
        for start in range(0, uv_query.shape[0], chunk_size):
            q = uv_query[start:start + chunk_size]
            diff = q.unsqueeze(1) - uv_face.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)
            mins.append(torch.norm(diff, dim=-1).min(dim=1).values)
        return torch.cat(mins, dim=0)

    @staticmethod
    def _estimate_uv_mask_tol(
        uv_face: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
        fallback: float = 0.05,
        scale: float = 2.5,
        max_points: int = 2048,
        chunk_size: int = 512,
    ) -> float:
        if uv_face.shape[0] < 2:
            return float(fallback)

        uv_cpu = uv_face.detach().to(device="cpu")
        n = uv_cpu.shape[0]
        if n > max_points:
            sample_idx = torch.linspace(0, n - 1, max_points).round().long()
            uv_cpu = uv_cpu[sample_idx]
            n = uv_cpu.shape[0]

        min_vals = []
        for start in range(0, n, chunk_size):
            q = uv_cpu[start:start + chunk_size]
            diff = q.unsqueeze(1) - uv_cpu.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)

            dist = torch.norm(diff, dim=-1)
            rows = q.shape[0]
            dist[torch.arange(rows), start:start + rows] = float("inf")
            min_vals.append(dist.min(dim=1).values)

        spacing = torch.cat(min_vals, dim=0).median()
        if not torch.isfinite(spacing):
            return float(fallback)
        return float(max(scale * float(spacing.item()), 1e-6))

    @staticmethod
    def _wrapped_grid_next(idx, size, periodic):
        nxt = idx + 1
        if periodic:
            return (idx + 1) % size
        if nxt >= size:
            return None
        return nxt
    def _make_density_image(self, face_tensors, rho_global, res=256):


        uv_all = []
        rho_all = []

        for ft in face_tensors:
            uv_i = ft["uv"].detach().cpu().numpy()                          # (Ni, 2)
            gidx_i = ft["global_vertex_idx"]                                # (Ni,)
            rho_i = rho_global[gidx_i].detach().cpu().numpy()               # (Ni,)

            uv_all.append(uv_i)
            rho_all.append(rho_i)

        uv_all = np.concatenate(uv_all, axis=0)
        rho_all = np.concatenate(rho_all, axis=0)

        gx, gy = np.meshgrid(
            np.linspace(0.0, 1.0, res),
            np.linspace(0.0, 1.0, res),
        )

        density_img = griddata(
            uv_all,
            rho_all,
            (gx, gy),
            method="linear",
            fill_value=0.0,
        )

        density_img = np.nan_to_num(density_img, nan=0.0, posinf=1.0, neginf=0.0)
        density_img = np.clip(density_img, 0.0, 1.0)
        return density_img
    def build_timelapse_render_cache(
        self,
        face_tensors,
        shape_or_path,
        grid_res_u=80,
        grid_res_v=80,
        uv_mask_tol=None,
        trim_tol=1e-7,
    ):
        cache = []

        for ft in face_tensors:
            device = ft["uv"].device
            dtype = ft["uv"].dtype
            uv_face = ft["uv"]
            u_periodic = bool(ft.get("u_periodic", False))
            v_periodic = bool(ft.get("v_periodic", False))

            uv_grid, _u_lin, _v_lin = self._build_face_uv_grid(ft, grid_res_u, grid_res_v)

            if uv_mask_tol is None:
                uv_mask_tol_i = self._estimate_uv_mask_tol(
                    uv_face=uv_face,
                    u_periodic=u_periodic,
                    v_periodic=v_periodic,
                )
            else:
                uv_mask_tol_i = uv_mask_tol

            dmin = self._periodic_uv_min_dist(
                uv_grid,
                uv_face,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
            mask_dense_prefilter = dmin <= uv_mask_tol_i
            uv_query = uv_grid[mask_dense_prefilter]

            geom = self.generator.eval_face_uv_from_face_tensor(
                shape_or_path=shape_or_path,
                face_tensor=ft,
                uv_norm=uv_query,
                metric_tol=getattr(self.generator, "metric_tol", 1e-9),
                trim_tol=trim_tol,
                as_torch=True,
            )

            valid_mask = geom["valid_mask"]
            uv_dense = geom["uv_norm"][valid_mask]
            xyz_dense = geom["points_xyz"][valid_mask]
            Xu_dense = geom["Xu"][valid_mask]
            Xv_dense = geom["Xv"][valid_mask]
            mask_dense_valid = torch.zeros_like(mask_dense_prefilter, dtype=torch.bool)
            mask_dense_valid[mask_dense_prefilter] = valid_mask

            local_face_id = torch.zeros(
                uv_dense.shape[0], dtype=torch.long, device=device
            )

            boundary_uv_i = None
            boundary_face_id_i = None
            true_bidx_i = self._true_open_boundary_idx(ft)
            if true_bidx_i.numel() > 0:
                boundary_uv_i = uv_face[true_bidx_i]
                boundary_face_id_i = torch.zeros(
                    boundary_uv_i.shape[0], dtype=torch.long, device=device
                )

            cache.append({
                "face_id": ft["face_id"],
                "uv_dense": uv_dense,
                "xyz_dense": xyz_dense,
                "Xu_dense": Xu_dense,
                "Xv_dense": Xv_dense,
                "local_face_id": local_face_id,
                "boundary_uv": boundary_uv_i,
                "boundary_face_id": boundary_face_id_i,
                "mask_dense_prefilter": mask_dense_prefilter,
                "mask_dense_valid": mask_dense_valid,
                "grid_shape": (grid_res_u, grid_res_v),
            })

        return cache
    def evaluate_cached_face_fields(self, render_cache, decoder, pred):
        decoder_out = decoder.evaluate_at_uv(
            points_uv=render_cache["uv_dense"],
            Xu=render_cache["Xu_dense"],
            Xv=render_cache["Xv_dense"],
            tau=self.cfg.tau,
            seeds_raw=pred["seeds_raw"],
            w_raw=pred["w_raw"],
            h_raw=pred.get("h_raw", None),
            theta=pred.get("theta", None),
            a_raw=pred.get("a_raw", None),
            points_face_id=render_cache["local_face_id"],
            boundary_uv=render_cache["boundary_uv"],
            boundary_face_id=render_cache["boundary_face_id"],
            boundary_width_raw=pred.get("boundary_width_raw", None),
            boundary_alpha_raw=None,
            boundary_beta_raw=None,
            seed_gates=pred.get("gate_probs", None),
        )

        return {
            "xyz_dense": render_cache["xyz_dense"],
            "rho_dense": decoder_out["rho"],
            "fiber3d_dense": decoder_out["fiber3d"],
            "grid_shape": render_cache["grid_shape"],
            "mask_dense_valid": render_cache["mask_dense_valid"],
        }

    @staticmethod
    def _build_faces_from_dense_mask(mask, Nu, Nv, rho_dense=None, thr=None):
        full_indices = -np.ones(mask.shape[0], dtype=int)
        full_indices[mask] = np.arange(mask.sum())

        faces = []

        def idx(i, j):
            return i * Nv + j

        for i in range(Nu - 1):
            for j in range(Nv - 1):
                ids = [idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)]
                mapped = [full_indices[k] for k in ids]
                if any(m < 0 for m in mapped):
                    continue

                i0, i1, i2, i3 = mapped

                if rho_dense is None or thr is None:
                    faces.append([3, i0, i1, i2])
                    faces.append([3, i2, i1, i3])
                    continue

                if rho_dense[i0] >= thr and rho_dense[i1] >= thr and rho_dense[i2] >= thr:
                    faces.append([3, i0, i1, i2])
                if rho_dense[i2] >= thr and rho_dense[i1] >= thr and rho_dense[i3] >= thr:
                    faces.append([3, i2, i1, i3])

        if len(faces) == 0:
            return None
        return np.array(faces, dtype=np.int64).reshape(-1)

    @staticmethod
    def _concat_polydata(meshes, scalar_name=None):
        if len(meshes) == 0:
            return None

        pts_parts = []
        face_parts = []
        scalar_parts = []
        offset = 0

        for mesh in meshes:
            pts = np.asarray(mesh.points, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4).copy()
            faces[:, 1:] += offset
            pts_parts.append(pts)
            face_parts.append(faces.reshape(-1))
            if scalar_name is not None:
                scalar_parts.append(np.asarray(mesh[scalar_name], dtype=np.float32))
            offset += pts.shape[0]

        out = pv.PolyData(
            np.concatenate(pts_parts, axis=0),
            np.concatenate(face_parts, axis=0),
        )
        if scalar_name is not None and len(scalar_parts) > 0:
            out[scalar_name] = np.concatenate(scalar_parts, axis=0)
        return out

    @staticmethod
    def _composite_to_white(img):
        if img.ndim != 3:
            return img
        if img.shape[2] == 3:
            return img
        if img.shape[2] != 4:
            return img[..., :3]

        rgb = img[..., :3].astype(np.float32)
        alpha = (img[..., 3:4].astype(np.float32) / 255.0)
        white = np.full_like(rgb, 255.0)
        out = rgb * alpha + white * (1.0 - alpha)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _render_offscreen_plotter(plotter, view_name):
        tight_view = None
        if view_name == "xy":
            plotter.enable_parallel_projection()
            plotter.view_xy()
            tight_view = "xy"
        elif view_name == "xz":
            plotter.enable_parallel_projection()
            plotter.view_xz()
            tight_view = "xz"
        elif view_name == "yz":
            plotter.enable_parallel_projection()
            plotter.view_yz()
            tight_view = "yz"
        else:
            plotter.disable_parallel_projection()
            plotter.view_isometric()
            tight_view = None

        plotter.reset_camera()
        if tight_view is not None:
            try:
                plotter.camera.tight(view=tight_view, adjust_render_window=False)
            except Exception:
                pass
            try:
                plotter.camera.zoom(0.90)
            except Exception:
                pass
        else:
            try:
                plotter.camera.zoom(0.94)
            except Exception:
                pass
        img = plotter.screenshot(return_img=True, transparent_background=False)
        return NN_Trainer._composite_to_white(img)

    @staticmethod
    def _add_image_title(img, title, pad=10, band_height=42):
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        title_band = np.full((band_height, img.shape[1], 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.72
        thickness = 2
        text_size, baseline = cv2.getTextSize(title, font, font_scale, thickness)
        x = max(pad, (img.shape[1] - text_size[0]) // 2)
        y = max(pad + text_size[1], (band_height + text_size[1]) // 2 - baseline)
        cv2.putText(
            title_band,
            title,
            (x, y),
            font,
            font_scale,
            (32, 32, 32),
            thickness,
            lineType=cv2.LINE_AA,
        )
        return np.vstack([title_band, img])

    @staticmethod
    def _add_panel_border(img, pad=10, border=2, bg_color=(255, 255, 255), border_color=(180, 186, 195)):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        inner = cv2.copyMakeBorder(
            img,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_color,
        )
        return cv2.copyMakeBorder(
            inner,
            border,
            border,
            border,
            border,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )

    @staticmethod
    def _pad_to_size(img, target_h=None, target_w=None, bg_color=(255, 255, 255)):
        h, w = img.shape[:2]
        if target_h is None:
            target_h = h
        if target_w is None:
            target_w = w
        if h == target_h and w == target_w:
            return img

        top = 0
        bottom = max(0, target_h - h)
        left = max(0, (target_w - w) // 2)
        right = max(0, target_w - w - left)
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_color,
        )

    @staticmethod
    def _resize_to_width(img, target_w):
        h, w = img.shape[:2]
        if w == target_w:
            return img
        target_h = max(1, int(round(h * (target_w / w))))
        return cv2.resize(img, (target_w, target_h))

    @staticmethod
    def _equalize_row_heights(images):
        target_h = min(img.shape[0] for img in images)
        out = []
        for img in images:
            h, w = img.shape[:2]
            if h == target_h:
                out.append(img)
            else:
                target_w = max(1, int(round(w * (target_h / h))))
                out.append(cv2.resize(img, (target_w, target_h)))
        return out

    @staticmethod
    def _stack_row_with_gaps(images, gap=18, bg_color=(255, 255, 255)):
        images = NN_Trainer._equalize_row_heights(images)
        if len(images) == 1:
            return images[0]
        gap_tile = np.full((images[0].shape[0], gap, 3), bg_color, dtype=np.uint8)
        parts = []
        for i, img in enumerate(images):
            parts.append(img)
            if i != len(images) - 1:
                parts.append(gap_tile)
        return np.hstack(parts)

    @staticmethod
    def _center_row_to_width(images, target_w, gap=18, bg_color=(255, 255, 255)):
        row = NN_Trainer._stack_row_with_gaps(images, gap=gap, bg_color=bg_color)
        if row.shape[1] > target_w:
            row = NN_Trainer._resize_to_width(row, target_w)
        return NN_Trainer._pad_to_size(row, target_w=target_w, bg_color=bg_color)

    @staticmethod
    def _stack_col_with_gaps(images, gap=22, bg_color=(255, 255, 255)):
        target_w = max(img.shape[1] for img in images)
        resized = [NN_Trainer._resize_to_width(img, target_w) for img in images]
        if len(resized) == 1:
            return resized[0]
        gap_tile = np.full((gap, target_w, 3), bg_color, dtype=np.uint8)
        parts = []
        for i, img in enumerate(resized):
            parts.append(img)
            if i != len(resized) - 1:
                parts.append(gap_tile)
        return np.vstack(parts)

    def _render_current_cad_frame_cached(
        self,
        seeds_list,
        decoders,
        pred_list,
        render_cache,
        thr=0.5,
    ):


        pred_by_face_id = {p["face_id"]: p for p in pred_list}
        dec_by_face_id = {ft["face_id"]: dec for ft, dec in zip(self.current_face_tensors, decoders)}
        density_meshes = []
        solid_meshes = []
        fiber_xyz_parts = []
        fiber_vec_parts = []
        fiber_rho_parts = []

        for cache_i in render_cache:
            face_id = cache_i["face_id"]
            pred = pred_by_face_id[face_id]
            decoder = dec_by_face_id[face_id]

            out = self.evaluate_cached_face_fields(cache_i, decoder, pred)

            xyz = out["xyz_dense"].detach().cpu().numpy()
            rho_dense = out["rho_dense"].detach().cpu().numpy()
            fiber_dense = out["fiber3d_dense"].detach().cpu().numpy()
            Nu, Nv = out["grid_shape"]
            mask = out["mask_dense_valid"].cpu().numpy()
            faces_all = self._build_faces_from_dense_mask(mask, Nu, Nv)
            if faces_all is not None:
                mesh_all = pv.PolyData(xyz, faces_all)
                mesh_all["rho"] = rho_dense.astype(np.float32)
                density_meshes.append(mesh_all)

            faces_solid = self._build_faces_from_dense_mask(mask, Nu, Nv, rho_dense=rho_dense, thr=thr)
            if faces_solid is not None:
                solid_meshes.append(pv.PolyData(xyz, faces_solid))

            valid_fiber = np.isfinite(rho_dense)
            valid_fiber &= np.isfinite(fiber_dense).all(axis=1)
            valid_fiber &= (rho_dense >= float(thr))
            valid_fiber &= (np.linalg.norm(fiber_dense, axis=1) > 1e-10)
            if np.any(valid_fiber):
                fiber_xyz_parts.append(xyz[valid_fiber])
                fiber_vec_parts.append(fiber_dense[valid_fiber])
                fiber_rho_parts.append(rho_dense[valid_fiber])

        if density_meshes:
            rho_all = np.concatenate([m["rho"] for m in density_meshes], axis=0)
            rho_clim = [0.0, max(1.0, float(np.quantile(rho_all, 0.995)))]
        else:
            rho_clim = [0.0, 1.0]

        density_mesh_merged = self._concat_polydata(density_meshes, scalar_name="rho")
        solid_mesh_merged = self._concat_polydata(solid_meshes, scalar_name=None)

        all_points = []
        for mesh in density_meshes:
            all_points.append(np.asarray(mesh.points))
        if all_points:
            all_points = np.concatenate(all_points, axis=0)
            diag = float(np.linalg.norm(np.ptp(all_points, axis=0)))
        else:
            diag = 1.0
        arrow_scale = 0.04 * max(diag, 1e-6)

        fiber_points = None
        fiber_vectors = None
        fiber_rho = None
        if fiber_xyz_parts:
            fiber_points = np.concatenate(fiber_xyz_parts, axis=0).astype(np.float32)
            fiber_vectors = np.concatenate(fiber_vec_parts, axis=0).astype(np.float32)
            fiber_rho = np.concatenate(fiber_rho_parts, axis=0).astype(np.float32)
            max_arrows = 600
            if fiber_points.shape[0] > max_arrows:
                stride = int(np.ceil(fiber_points.shape[0] / max_arrows))
                fiber_points = fiber_points[::stride]
                fiber_vectors = fiber_vectors[::stride]
                fiber_rho = fiber_rho[::stride]

        seed_vis = self._seed_points_xyz_and_gates_all_faces(
            seeds_list=seeds_list,
            pred_list=pred_list,
            face_tensors=self.current_face_tensors,
        )
        active_seed_points = seed_vis["xyz_active"]
        inactive_seed_points = seed_vis["xyz_inactive"]
        seed_point_size = max(6.0, 0.006 * max(diag, 1.0) * 100.0)
        show_seed_points = True
        show_axes_widget = True

        def make_plotter(title, mode, window_size):
            pl = pv.Plotter(off_screen=True, window_size=window_size)
            pl.set_background("white")
            try:
                pl.disable_anti_aliasing()
            except Exception:
                pass
            try:
                pl.ren_win.SetMultiSamples(0)
            except Exception:
                pass
            pl.remove_all_lights()

            if mode == "density":
                if density_mesh_merged is not None:
                    pl.add_mesh(
                        density_mesh_merged,
                        scalars="rho",
                        cmap="viridis",
                        clim=rho_clim,
                        show_edges=False,
                        lighting=False,
                        smooth_shading=False,
                        nan_color="white",
                        interpolate_before_map=False,
                        scalar_bar_args={
                            "title": "rho",
                            "position_x": 0.28,
                            "position_y": 0.02,
                            "width": 0.64,
                            "height": 0.05,
                            "title_font_size": 12,
                            "label_font_size": 10,
                            "color": "#4b5563",
                            "fmt": "%.2f",
                            "n_labels": 5,
                        },
                    )
            elif mode == "solid":
                if solid_mesh_merged is not None:
                    pl.add_mesh(
                        solid_mesh_merged,
                        color="#8ecae6",
                        smooth_shading=False,
                        specular=0.0,
                        show_edges=False,
                        lighting=False,
                    )
            elif mode == "fiber":
                if solid_mesh_merged is not None:
                    pl.add_mesh(
                        solid_mesh_merged,
                        color="#dbeafe",
                        opacity=1.0,
                        smooth_shading=False,
                        show_edges=False,
                        lighting=False,
                    )
                if fiber_points is not None and fiber_points.shape[0] > 0:
                    cloud = pv.PolyData(fiber_points)
                    cloud["vectors"] = fiber_vectors
                    cloud["rho"] = fiber_rho
                    glyphs = cloud.glyph(
                        orient="vectors",
                        scale=False,
                        factor=arrow_scale,
                        geom=pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0)),
                    )
                    pl.add_mesh(glyphs, color="#1d4ed8", line_width=2)

            if show_seed_points and active_seed_points is not None and len(active_seed_points) > 0:
                pl.add_mesh(
                    pv.PolyData(active_seed_points.astype(np.float32)),
                    color="red",
                    render_points_as_spheres=True,
                    point_size=seed_point_size,
                )
            if show_seed_points and inactive_seed_points is not None and len(inactive_seed_points) > 0:
                pl.add_mesh(
                    pv.PolyData(inactive_seed_points.astype(np.float32)),
                    color="gray",
                    opacity=0.35,
                    render_points_as_spheres=True,
                    point_size=max(5.0, 0.8 * seed_point_size),
                )
            if show_axes_widget:
                pl.show_axes()
            return pl

        top_specs = [
            ("Material Distribution | Front View", "density", "xz"),
            ("Material Distribution | Side View", "density", "yz"),
            ("Material Distribution | Top View", "density", "xy"),
        ]
        bottom_specs = [
            ("Material Distribution | Perspective View", "density", "iso"),
            (f"Hollowed Model | Perspective View | thr={thr:.2f}", "solid", "iso"),
        ]

        top_imgs = []
        for title, mode, view in top_specs:
            pl = make_plotter(title, mode, window_size=(520, 280))
            img = self._render_offscreen_plotter(pl, view)
            top_imgs.append(self._add_panel_border(self._add_image_title(img, title)))
            pl.close()

        bottom_imgs = []
        for title, mode, view in bottom_specs:
            pl = make_plotter(title, mode, window_size=(820, 820))
            img = self._render_offscreen_plotter(pl, view)
            bottom_imgs.append(self._add_panel_border(self._add_image_title(img, title)))
            pl.close()

        col_gap = 22
        row_gap = 28
        top_row = self._stack_row_with_gaps(top_imgs, gap=col_gap)
        bottom_row = self._center_row_to_width(bottom_imgs, target_w=top_row.shape[1], gap=col_gap)
        gap_tile = np.full((row_gap, top_row.shape[1], 3), 255, dtype=np.uint8)
        cad_panel = np.vstack([top_row, gap_tile, bottom_row])
        cad_panel = cv2.copyMakeBorder(
            cad_panel,
            16,
            16,
            16,
            16,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        return cad_panel
    def _gate_activity_stats(
    self,
    gate_probs: torch.Tensor | None,
    threshold: float,
) -> dict[str, float]:
        if gate_probs is None:
            return {
                "active_count": 0.0,
                "active_frac": 0.0,
                "gate_min": 0.0,
                "gate_mean": 0.0,
                "gate_max": 0.0,
            }

        g = gate_probs.detach()
        active = (g > threshold)

        return {
            "active_count": float(active.sum().item()),
            "active_frac": float(active.float().mean().item()),
            "gate_min": float(g.min().item()),
            "gate_mean": float(g.mean().item()),
            "gate_max": float(g.max().item()),
        }

    @staticmethod
    def _gate_stats(gates: torch.Tensor | None) -> dict[str, float]:
        if gates is None:
            return {
                "min": 0.0,
                "mean": 0.0,
                "max": 0.0,
            }

        g = gates.detach()
        return {
            "min": float(g.min().item()),
            "mean": float(g.mean().item()),
            "max": float(g.max().item()),
        }

    def _use_hard_gate_mask(self, step: int, steps_since_improve: int) -> bool:
        cfg = self.cfg
        if not getattr(cfg, "use_hard_gate_mask", False):
            return False

        start_step = int(round(float(cfg.hard_gate_start_frac) * float(cfg.num_steps)))
        if step >= start_step:
            return True

        if not getattr(cfg, "hard_gate_allow_early_stability", True):
            return False

        min_step = int(round(float(cfg.hard_gate_min_frac_for_stability) * float(cfg.num_steps)))
        if step < min_step:
            return False

        return steps_since_improve >= int(cfg.hard_gate_stability_patience)

    def _decoder_gate_values(
        self,
        gates_struct: torch.Tensor | None,
        use_hard_gate_mask: bool,
        threshold: float,
    ) -> torch.Tensor | None:
        if gates_struct is None:
            return None

        if not use_hard_gate_mask:
            return gates_struct

        active = (gates_struct >= threshold)
        if not bool(active.any()):
            active = torch.zeros_like(active, dtype=torch.bool)
            active[torch.argmax(gates_struct)] = True

        return active.to(dtype=gates_struct.dtype)

    def _seed_points_xyz_and_gates_all_faces(self, seeds_list, pred_list, face_tensors):
        xyz_active = []
        xyz_inactive = []
        gate_active = []
        gate_inactive = []

        thr = self.cfg.Gating_binirize_limit

        for seeds, pred, ft in zip(seeds_list, pred_list, face_tensors):
            xyz_i = self.generator.seeds_uv_to_xyz_nearest(
                seeds,
                ft["uv"],
                ft["points_xyz"],
            )

            gates_i = pred.get("gate_probs", None)

            if gates_i is None:
                xyz_active.append(xyz_i)
                continue

            g = gates_i.detach().cpu()
            active_mask = (g > thr).cpu().numpy()

            xyz_i_active = xyz_i[active_mask]
            xyz_i_inactive = xyz_i[~active_mask]

            if len(xyz_i_active) > 0:
                xyz_active.append(xyz_i_active)
                gate_active.append(g[active_mask].numpy())

            if len(xyz_i_inactive) > 0:
                xyz_inactive.append(xyz_i_inactive)
                gate_inactive.append(g[~active_mask].numpy())

        import numpy as np

        xyz_active = np.concatenate(xyz_active, axis=0) if len(xyz_active) > 0 else None
        xyz_inactive = np.concatenate(xyz_inactive, axis=0) if len(xyz_inactive) > 0 else None
        gate_active = np.concatenate(gate_active, axis=0) if len(gate_active) > 0 else None
        gate_inactive = np.concatenate(gate_inactive, axis=0) if len(gate_inactive) > 0 else None

        return {
            "xyz_active": xyz_active,
            "xyz_inactive": xyz_inactive,
            "gate_active": gate_active,
            "gate_inactive": gate_inactive,
        }
    
    @staticmethod
    def sharpen_gate_probs(gates: torch.Tensor | None, Gating_binirize_sharpness:float = 20.0,Gating_binirize_limit:float = 0.45,eps: float = 1e-8):
        if gates is None:
            return None
        g_raw = gates.clamp(eps, 1.0 - eps)
        g_soft = torch.sigmoid(Gating_binirize_sharpness* (g_raw - Gating_binirize_limit))
        return g_soft
    def visualize_best_seed_activity(self, result, points_xyz=None, faces_ijk=None):
        best_seeds = result["best_seeds"]
        best_pred = result["best_pred"]
        face_tensors = result["face_tensors"]

        seed_vis = self._seed_points_xyz_and_gates_all_faces(
            seeds_list=best_seeds,
            pred_list=best_pred,
            face_tensors=face_tensors,
        )

        plotter = pv.Plotter()

        if points_xyz is not None and faces_ijk is not None:
            pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)
            mesh = pv.PolyData(points_xyz.detach().cpu().numpy(), pv_faces_fixed)
            plotter.add_mesh(mesh, color="white", opacity=0.25, show_edges=False)

        if seed_vis["xyz_active"] is not None and len(seed_vis["xyz_active"]) > 0:
            active_cloud = pv.PolyData(seed_vis["xyz_active"])
            plotter.add_mesh(
                active_cloud,
                color="red",
                render_points_as_spheres=True,
                point_size=14,
                label="Active seeds",
            )

        if seed_vis["xyz_inactive"] is not None and len(seed_vis["xyz_inactive"]) > 0:
            inactive_cloud = pv.PolyData(seed_vis["xyz_inactive"])
            plotter.add_mesh(
                inactive_cloud,
                color="gray",
                render_points_as_spheres=True,
                point_size=10,
                opacity=0.4,
                label="Vanished seeds",
            )

        plotter.add_legend()
        plotter.show()
    

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_result_stepwise(self, result, points_xyz, faces_ijk):
        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        density_init = result["Initial_shape_density"].detach().cpu().numpy()
        density_mid = result["Mid_shape_density"].detach().cpu().numpy()
        density_final = result["Final_shape_density"].detach().cpu().numpy()

        self.viz.plot_density_and_seedpoints_3stage_2(
            mesh_points=points_xyz.detach().cpu().numpy(),
            pv_faces=pv_faces_fixed,
            density_init=density_init,
            density_mid=density_mid,
            density_final=density_final,
            seed_points_init=result["seed_points_init"],
            seed_points_mid=result["seed_points_mid"],
            seed_points_final=result["seed_points_final"],
        )

    def visualize_result_final(self, result, points_xyz, faces_ijk, thr=0.5, show_solid=True):
        density_fin_viz = self.viz.viz_normalize(result["Final_shape_density"])
        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        solid, thr_used, _ = self.viz.visualize_density_thresholded(
            points=points_xyz,
            pv_faces=pv_faces_fixed,
            density_total=density_fin_viz,
            thr=thr,
            show_solid=show_solid,
        )
        return solid, thr_used
    def sample_face_field_for_visualization(
        self,
        ft: dict,
        decoder,
        pred: dict,
        shape_or_path,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        use_boundary_attachment: bool = True,
        trim_tol: float = 1e-7,
    ):
        """
        Dense CAD-native field sampling on one face for smooth visualization.

        This version:
        - builds a dense UV grid in normalized face UV
        - optionally prefilters points by proximity to sampled UV cloud
        - evaluates xyz, Xu, Xv on the actual CAD face
        - keeps only trim-valid points
        - evaluates decoder on those dense query points

        Returns:
            {
                "uv_dense": (Nd,2),
                "uv_raw_dense": (Nd,2),
                "xyz_dense": (Nd,3),
                "Xu_dense": (Nd,3),
                "Xv_dense": (Nd,3),
                "rho_dense": (Nd,),
                "rho_v_dense": (Nd,),
                "rho_b_dense": (Nd,),
                "fiber3d_dense": (Nd,3),
                "edge_field_dense": (Nd,),
                "mask_dense_prefilter": (Nu*Nv,),
                "grid_shape": (Nu, Nv),
            }
        """
        device = ft["uv"].device
        dtype = ft["uv"].dtype

        uv_face = ft["uv"]
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        # ------------------------------------------------------------
        # 1) Dense UV grid in normalized face UV coordinates
        # ------------------------------------------------------------
        uv_grid, _u_lin, _v_lin = self._build_face_uv_grid(ft, grid_res_u, grid_res_v)

        # ------------------------------------------------------------
        # 2) Optional UV-cloud prefilter
        #    Helps avoid querying huge empty regions on trimmed faces.
        # ------------------------------------------------------------
        if uv_mask_tol is None:
            uv_mask_tol = self._estimate_uv_mask_tol(
                uv_face=uv_face,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )

        dmin = self._periodic_uv_min_dist(
            uv_grid,
            uv_face,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        mask_dense_prefilter = dmin <= uv_mask_tol
        uv_query = uv_grid[mask_dense_prefilter]

        if uv_query.numel() == 0:
            raise ValueError(
                f"No dense UV query points survived prefilter on face {ft.get('face_id', 'unknown')}. "
                f"Try increasing uv_mask_tol."
            )

        # ------------------------------------------------------------
        # 3) CAD-native geometry evaluation
        # ------------------------------------------------------------
        geom = self.generator.eval_face_uv_from_face_tensor(
            shape_or_path=shape_or_path,
            face_tensor=ft,
            uv_norm=uv_query,
            metric_tol=getattr(self.generator, "metric_tol", 1e-9),
            trim_tol=trim_tol,
            as_torch=True,
        )

        valid_mask = geom["valid_mask"]
        if valid_mask.numel() == 0 or not bool(valid_mask.any().item()):
            raise ValueError(
                f"No valid CAD-evaluable dense points on face {ft.get('face_id', 'unknown')}."
            )

        uv_dense = geom["uv_norm"][valid_mask]
        uv_raw_dense = geom["uv_raw"][valid_mask]
        xyz_dense = geom["points_xyz"][valid_mask]
        Xu_dense = geom["Xu"][valid_mask]
        Xv_dense = geom["Xv"][valid_mask]
        mask_dense_valid = torch.zeros_like(mask_dense_prefilter, dtype=torch.bool)
        mask_dense_valid[mask_dense_prefilter] = valid_mask

        # ------------------------------------------------------------
        # 4) Boundary data for decoder
        # ------------------------------------------------------------
        local_face_id = torch.zeros(
            uv_dense.shape[0],
            dtype=torch.long,
            device=device,
        )

        boundary_uv_i = None
        boundary_face_id_i = None

        if use_boundary_attachment:
            true_bidx_i = self._true_open_boundary_idx(ft)
            if true_bidx_i.numel() > 0:
                boundary_uv_i = uv_face[true_bidx_i]
                boundary_face_id_i = torch.zeros(
                    boundary_uv_i.shape[0],
                    dtype=torch.long,
                    device=device,
                )

        # ------------------------------------------------------------
        # 5) Recover trained parameters
        # ------------------------------------------------------------
        seeds_raw = pred["seeds_raw"]
        w_raw = pred["w_raw"]
        h_raw = pred.get("h_raw", None)

        theta = pred.get("theta", None)
        a_raw = pred.get("a_raw", None)

        gates_i = pred.get("gate_probs", None)
        boundary_width_raw = pred.get("boundary_width_raw", None)
        boundary_alpha_raw = pred.get("boundary_alpha_raw", None)
        boundary_beta_raw = pred.get("boundary_beta_raw", None)

        # ------------------------------------------------------------
        # 6) Evaluate decoder on CAD-native dense query points
        # ------------------------------------------------------------
        decoder_out = decoder.evaluate_at_uv(
            points_uv=uv_dense,
            Xu=Xu_dense,
            Xv=Xv_dense,
            tau=self.cfg.tau,
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            h_raw=h_raw,
            theta=theta,
            a_raw=a_raw,
            points_face_id=local_face_id,
            boundary_uv=boundary_uv_i,
            boundary_face_id=boundary_face_id_i,
            boundary_width_raw=boundary_width_raw,
            boundary_alpha_raw=None,
            boundary_beta_raw=None,
            seed_gates=gates_i,
        )

        self._require_decoder_keys(
            decoder_out,
            ["rho", "rho_v", "rho_b", "fiber3d", "edge_field"],
        )

        return {
            "face_id": ft["face_id"],
            "uv_dense": uv_dense,
            "uv_raw_dense": uv_raw_dense,
            "xyz_dense": xyz_dense,
            "Xu_dense": Xu_dense,
            "Xv_dense": Xv_dense,
            "rho_dense": decoder_out["rho"],
            "rho_v_dense": decoder_out["rho_v"],
            "rho_b_dense": decoder_out["rho_b"],
            "fiber3d_dense": decoder_out["fiber3d"],
            "edge_field_dense": decoder_out["edge_field"],
            "mask_dense_prefilter": mask_dense_prefilter,
            "mask_dense_valid": mask_dense_valid,
            "grid_shape": (grid_res_u, grid_res_v),
        }
   
    def sample_result_field_dense_for_visualization(
        self,
        result: dict,
        shape_or_path=None,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        use_best_pred: bool = True,
    ):
        """
        Dense CAD-native field sampling over all faces for smooth visualization.
        """
        face_tensors = result["face_tensors"]
        decoders = result["decoders"]

        if use_best_pred:
            pred_list = result["best_pred"]
        else:
            raise ValueError("Only use_best_pred=True is currently supported.")

        if shape_or_path is None:
            shape_or_path = result.get("shape_path", None)

        if shape_or_path is None:
            raise ValueError(
                "shape_or_path is required for CAD-native dense sampling. "
                "Pass it explicitly or store 'shape_path' in result."
            )

        pred_by_face_id = {p["face_id"]: p for p in pred_list}

        xyz_parts = []
        rho_parts = []
        rho_v_parts = []
        rho_b_parts = []
        fiber_parts = []
        edge_parts = []
        face_ranges = []
        per_face = []

        start = 0
        for ft, decoder in zip(face_tensors, decoders):
            face_id = ft["face_id"]
            if face_id not in pred_by_face_id:
                raise KeyError(f"Missing best_pred for face_id={face_id}")

            pred = pred_by_face_id[face_id]

            sampled = self.sample_face_field_for_visualization(
                ft=ft,
                decoder=decoder,
                pred=pred,
                shape_or_path=shape_or_path,
                grid_res_u=grid_res_u,
                grid_res_v=grid_res_v,
                uv_mask_tol=uv_mask_tol,
            )

            n = sampled["xyz_dense"].shape[0]
            end = start + n

            xyz_parts.append(sampled["xyz_dense"])
            rho_parts.append(sampled["rho_dense"])
            rho_v_parts.append(sampled["rho_v_dense"])
            rho_b_parts.append(sampled["rho_b_dense"])
            fiber_parts.append(sampled["fiber3d_dense"])
            edge_parts.append(sampled["edge_field_dense"])

            face_ranges.append((start, end, face_id))
            per_face.append(sampled)
            start = end

        return {
            "points_xyz": torch.cat(xyz_parts, dim=0),
            "rho": torch.cat(rho_parts, dim=0),
            "rho_v": torch.cat(rho_v_parts, dim=0),
            "rho_b": torch.cat(rho_b_parts, dim=0),
            "fiber3d": torch.cat(fiber_parts, dim=0),
            "edge_field": torch.cat(edge_parts, dim=0),
            "face_ranges": face_ranges,
            "per_face": per_face,
        }

    @staticmethod
    def _resolve_visualization_grid_resolution(
        grid_res_u: int,
        grid_res_v: int,
        dense_factor: float = 1.0,
        min_res: int = 8,
        max_res: int = 1024,
    ) -> tuple[int, int]:
        dense_factor = float(max(dense_factor, 1e-3))
        res_u = int(round(float(grid_res_u) * dense_factor))
        res_v = int(round(float(grid_res_v) * dense_factor))
        res_u = max(int(min_res), min(int(max_res), res_u))
        res_v = max(int(min_res), min(int(max_res), res_v))
        return res_u, res_v

    def visualize_result_final_smooth_points(
        self,
        result,
        shape_or_path=None,
        thr: float = 0.5,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        dense_factor: float = 1.0,
    ):
        """
        Smooth point-cloud style threshold visualization from dense CAD-native decoder sampling.

        `dense_factor` scales the internal UV sampling density used for visualization.
        Larger values produce a denser point cloud and finer visual detail.
        """
        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )

        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

        points_xyz = dense["points_xyz"].detach().cpu().numpy()
        rho = dense["rho"].detach().cpu().numpy()

        keep = rho >= thr
        solid_points = points_xyz[keep]

        print(
            f"Smooth CAD-native visualization: kept {keep.sum()} / {keep.shape[0]} dense points "
            f"with threshold {thr:.3f} on grid ({grid_res_u} x {grid_res_v})"
        )


        cloud = pv.PolyData(solid_points)

        plotter = pv.Plotter()
        plotter.add_points(
            cloud,
            render_points_as_spheres=True,
            point_size=6,
        )

        plotter.show()

        return {
            "solid_points": solid_points,
            "points_xyz": points_xyz,
            "rho": rho,
            "keep_mask": keep,
            "dense": dense,
        }

    def Visualize_fresult_final_fiber_Direction(
        self,
        result,
        points_xyz,
        faces_ijk,
        thr: float = 0.5,
    ):
        import numpy as np
        import pyvista as pv

        if result.get("Final_shape_fiber_direction", None) is None:
            raise ValueError(
                "result['Final_shape_fiber_direction'] is missing. "
                "Run training with the updated trainer result output."
            )

        density = result["Final_shape_density"].detach().cpu()
        fiber = result["Final_shape_fiber_direction"].detach().cpu()
        points_xyz_cpu = points_xyz.detach().cpu()
        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        keep = torch.isfinite(density)
        keep = keep & torch.isfinite(fiber).all(dim=1)
        keep = keep & (density >= float(thr))
        keep = keep & (torch.linalg.norm(fiber, dim=1) > 1e-10)

        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if keep_idx.numel() == 0:
            print(f"No fiber arrows to display for threshold {thr:.3f}.")
            return {
                "points_xyz": points_xyz_cpu.numpy(),
                "rho": density.numpy(),
                "fiber3d": fiber.numpy(),
                "keep_mask": keep.numpy(),
            }

        max_arrows = 2000
        if keep_idx.numel() > max_arrows:
            step = int(np.ceil(float(keep_idx.numel()) / float(max_arrows)))
            keep_idx = keep_idx[::step]

        pts_np = points_xyz_cpu[keep_idx].numpy().astype(np.float32)
        fiber_np = fiber[keep_idx].numpy().astype(np.float32)
        rho_np = density[keep_idx].numpy().astype(np.float32)

        bbox = points_xyz_cpu.amax(dim=0) - points_xyz_cpu.amin(dim=0)
        diag = float(torch.linalg.norm(bbox).item())
        arrow_scale_used = 0.03 * diag

        plotter = pv.Plotter()

        surface = pv.PolyData(
            points_xyz_cpu.numpy().astype(np.float32),
            pv_faces_fixed,
        )
        surface["rho"] = density.numpy().astype(np.float32)
        plotter.add_mesh(
            surface,
            scalars="rho",
            cmap="Greys",
            opacity=0.20,
            show_edges=False,
        )

        arrow_cloud = pv.PolyData(pts_np)
        arrow_cloud["vectors"] = fiber_np
        arrow_cloud["rho"] = rho_np

        glyphs = arrow_cloud.glyph(
            orient="vectors",
            scale=False,
            factor=arrow_scale_used,
            geom=pv.Arrow(),
        )
        plotter.add_mesh(glyphs, color="royalblue")
        plotter.show_axes()
        plotter.show()

        print(
            f"Fiber-direction visualization: showing {pts_np.shape[0]} arrows "
            f"with threshold {thr:.3f}"
        )

        return {
            "arrow_points": pts_np,
            "arrow_vectors": fiber_np,
            "arrow_rho": rho_np,
            "points_xyz": points_xyz_cpu.numpy(),
            "rho": density.numpy(),
            "fiber3d": fiber.numpy(),
            "keep_mask": keep.numpy(),
            "arrow_scale_used": float(arrow_scale_used),
        }

    def visualize_result_final_fiber_direction(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    def Visualize_fresult_final_fiber_Direction_2D(
        self,
        result,
        shape_or_path=None,
        thr: float = 0.5,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        dense_factor: float = 1.0,
        max_arrows_per_face: int = 1200,
        arrow_scale: float = 28.0,
        arrow_width: float = 0.0025,
        cmap: str = "viridis",
        show_boundary: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        if result.get("Final_shape_fiber_direction", None) is None:
            raise ValueError(
                "result['Final_shape_fiber_direction'] is missing. "
                "Run training with the updated trainer result output."
            )

        density_global = result["Final_shape_density"].detach().cpu()
        fiber_global = result["Final_shape_fiber_direction"].detach().cpu()
        face_tensors = result["face_tensors"]

        n_faces = len(face_tensors)
        ncols = min(3, max(1, n_faces))
        nrows = int(np.ceil(n_faces / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.5 * ncols, 5.0 * nrows),
            squeeze=False,
        )

        plotted_faces = []

        for ax, ft in zip(axes.ravel(), face_tensors):
            face_id = ft["face_id"]
            gidx = ft["global_vertex_idx"].detach().cpu()

            uv_face = ft["uv"].detach().cpu().numpy().astype(np.float32)
            Xu_face = ft["Xu"].detach().cpu().numpy().astype(np.float32)
            Xv_face = ft["Xv"].detach().cpu().numpy().astype(np.float32)
            rho_face = density_global[gidx].numpy().astype(np.float32)
            fiber_face = fiber_global[gidx].numpy().astype(np.float32)

            t_uv_face = self._fiber3d_to_uv_direction(
                Xu_np=Xu_face,
                Xv_np=Xv_face,
                fiber_np=fiber_face,
            )

            keep = (
                np.isfinite(rho_face)
                & np.isfinite(t_uv_face).all(axis=1)
                & (rho_face >= float(thr))
                & (np.linalg.norm(t_uv_face, axis=1) > 1e-10)
            )

            arrow_points = 0
            if np.count_nonzero(keep) > 0:
                uv_keep = uv_face[keep]
                rho_keep = rho_face[keep]
                t_uv_keep = t_uv_face[keep]

                if uv_keep.shape[0] > int(max_arrows_per_face):
                    pick = np.linspace(
                        0,
                        uv_keep.shape[0] - 1,
                        num=int(max_arrows_per_face),
                    ).round().astype(np.int64)
                    uv_keep = uv_keep[pick]
                    rho_keep = rho_keep[pick]
                    t_uv_keep = t_uv_keep[pick]

                ax.quiver(
                    uv_keep[:, 0],
                    uv_keep[:, 1],
                    t_uv_keep[:, 0],
                    t_uv_keep[:, 1],
                    rho_keep,
                    cmap=cmap,
                    clim=(0.0, 1.0),
                    angles="xy",
                    scale_units="xy",
                    scale=float(arrow_scale),
                    width=float(arrow_width),
                )
                arrow_points = int(uv_keep.shape[0])

            if show_boundary:
                bidx = self._true_open_boundary_idx(ft)
                if bidx.numel() > 0:
                    boundary_uv = ft["uv"][bidx].detach().cpu().numpy().astype(np.float32)
                    ax.scatter(
                        boundary_uv[:, 0],
                        boundary_uv[:, 1],
                        s=12,
                        c="red",
                        alpha=0.85,
                    )

            ax.set_title(f"Face {face_id} | arrows={arrow_points}")
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_aspect("equal")
            plotted_faces.append(face_id)

        for ax in axes.ravel()[len(face_tensors):]:
            ax.axis("off")

        fig.suptitle(
            f"Final fiber direction in UV domain on training points | thr={float(thr):.3f}",
            y=0.98,
        )
        fig.tight_layout()
        plt.show()

        print(
            f"2D fiber-direction visualization: plotted {len(plotted_faces)} faces "
            f"with threshold {thr:.3f}"
        )

        return {
            "figure": fig,
            "face_ids": plotted_faces,
            "thr_used": float(thr),
            "uv_by_face": {
                int(ft["face_id"]): ft["uv"].detach().cpu().numpy().astype(np.float32)
                for ft in face_tensors
            },
        }

    def visualize_result_final_fiber_direction_2d(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction_2D(*args, **kwargs)

    def _trace_fiber_path_on_vertices(
        self,
        start_idx,
        adjacency,
        points_np,
        fiber_np,
        keep_mask_np,
        sign=1.0,
        max_steps=80,
        min_align=0.15,
    ):
        path = [int(start_idx)]
        used = {int(start_idx)}
        prev = None

        for _ in range(int(max_steps)):
            cur = path[-1]
            cur_dir = fiber_np[cur]
            cur_nrm = np.linalg.norm(cur_dir)
            if cur_nrm <= 1e-12:
                break

            cur_dir = (float(sign) * cur_dir) / cur_nrm
            best_j = None
            best_score = -np.inf

            for nb in adjacency[cur]:
                nb = int(nb)
                if not keep_mask_np[nb]:
                    continue
                if prev is not None and nb == prev:
                    continue
                if nb in used:
                    continue

                edge = points_np[nb] - points_np[cur]
                edge_nrm = np.linalg.norm(edge)
                if edge_nrm <= 1e-12:
                    continue

                edge_hat = edge / edge_nrm
                score = float(np.dot(cur_dir, edge_hat))
                if score > best_score:
                    best_score = score
                    best_j = nb

            if best_j is None or best_score < float(min_align):
                break

            prev = cur
            path.append(best_j)
            used.add(best_j)

        return path

    @staticmethod
    def _fiber3d_to_uv_direction(Xu_np, Xv_np, fiber_np, eps=1e-12):
        a11 = np.sum(Xu_np * Xu_np, axis=1)
        a12 = np.sum(Xu_np * Xv_np, axis=1)
        a22 = np.sum(Xv_np * Xv_np, axis=1)
        b1 = np.sum(Xu_np * fiber_np, axis=1)
        b2 = np.sum(Xv_np * fiber_np, axis=1)
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < eps, np.nan, det)

        du = (a22 * b1 - a12 * b2) / det
        dv = (-a12 * b1 + a11 * b2) / det
        tuv = np.stack([du, dv], axis=1)
        nrm = np.linalg.norm(tuv, axis=1, keepdims=True)
        ok = np.isfinite(tuv).all(axis=1, keepdims=True) & (nrm > eps)
        tuv = np.where(ok, tuv / np.clip(nrm, eps, None), 0.0)
        return tuv.astype(np.float32)

    def _trace_streamline_on_dense_face(
        self,
        uv_np,
        xyz_np,
        t_uv_np,
        rho_np,
        start_idx,
        step_uv,
        max_steps,
        kdtree,
        align_thresh=0.2,
        min_rho=0.0,
        sign=1.0,
    ):
        path_xyz = [xyz_np[int(start_idx)]]
        path_uv = [uv_np[int(start_idx)]]
        current_uv = uv_np[int(start_idx)].copy()
        prev_dir = None

        for _ in range(int(max_steps)):
            _dist, idx = kdtree.query(current_uv, k=1)
            idx = int(idx)
            if rho_np[idx] < float(min_rho):
                break

            direction = t_uv_np[idx].astype(np.float64) * float(sign)
            nrm = np.linalg.norm(direction)
            if not np.isfinite(nrm) or nrm <= 1e-12:
                break
            direction = direction / nrm

            if prev_dir is not None and float(np.dot(direction, prev_dir)) < 0.0:
                direction = -direction

            next_uv = current_uv + float(step_uv) * direction
            dist_next, next_idx = kdtree.query(next_uv, k=1)
            next_idx = int(next_idx)
            if not np.isfinite(dist_next):
                break
            if next_idx == idx:
                break

            chord = uv_np[next_idx] - current_uv
            chord_nrm = np.linalg.norm(chord)
            if chord_nrm <= 1e-12:
                break
            align = float(np.dot(direction, chord / chord_nrm))
            if align < float(align_thresh):
                break

            current_uv = uv_np[next_idx].copy()
            prev_dir = direction
            path_uv.append(current_uv)
            path_xyz.append(xyz_np[next_idx])

        return np.asarray(path_uv, dtype=np.float32), np.asarray(path_xyz, dtype=np.float32)

    def Visualize_fresult_final_fiber_Paths(
        self,
        result,
        points_xyz,
        faces_ijk,
        thr: float = 0.5,
        shape_or_path=None,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        max_paths: int = 120,
        max_steps: int = 80,
        min_align: float = 0.20,
        line_width: float = 3.0,
        dense_factor: float = 1.0,
    ):
        import numpy as np
        import pyvista as pv
        from scipy.spatial import cKDTree

        if result.get("Final_shape_fiber_direction", None) is None:
            raise ValueError(
                "result['Final_shape_fiber_direction'] is missing. "
                "Run training with the updated trainer result output."
            )

        density = result["Final_shape_density"].detach().cpu()
        fiber = result["Final_shape_fiber_direction"].detach().cpu()
        points_xyz_cpu = points_xyz.detach().cpu()
        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        keep = torch.isfinite(density)
        keep = keep & torch.isfinite(fiber).all(dim=1)
        keep = keep & (density >= float(thr))
        keep = keep & (torch.linalg.norm(fiber, dim=1) > 1e-10)

        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if keep_idx.numel() == 0:
            print(f"No fiber paths to display for threshold {thr:.3f}.")
            return {
                "points_xyz": points_xyz_cpu.numpy(),
                "rho": density.numpy(),
                "fiber3d": fiber.numpy(),
                "keep_mask": keep.numpy(),
                "paths_vertex_idx": [],
            }

        density_np = density.numpy().astype(np.float32)
        points_np = points_xyz_cpu.numpy().astype(np.float32)

        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )
        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

        paths_xyz = []
        path_rho = []

        for face_data in dense["per_face"]:
            uv_face = face_data["uv_dense"].detach().cpu().numpy().astype(np.float32)
            xyz_face = face_data["xyz_dense"].detach().cpu().numpy().astype(np.float32)
            Xu_face = face_data["Xu_dense"].detach().cpu().numpy().astype(np.float32)
            Xv_face = face_data["Xv_dense"].detach().cpu().numpy().astype(np.float32)
            fiber_face = face_data["fiber3d_dense"].detach().cpu().numpy().astype(np.float32)
            rho_face = face_data["rho_dense"].detach().cpu().numpy().astype(np.float32)

            keep_face = (
                np.isfinite(rho_face)
                & np.isfinite(fiber_face).all(axis=1)
                & (rho_face >= float(thr))
                & (np.linalg.norm(fiber_face, axis=1) > 1e-10)
            )
            valid_idx = np.flatnonzero(keep_face)
            if valid_idx.size < 2:
                continue

            uv_keep = uv_face[keep_face]
            xyz_keep = xyz_face[keep_face]
            rho_keep = rho_face[keep_face]
            t_uv_keep = self._fiber3d_to_uv_direction(
                Xu_np=Xu_face[keep_face],
                Xv_np=Xv_face[keep_face],
                fiber_np=fiber_face[keep_face],
            )
            good_t = np.linalg.norm(t_uv_keep, axis=1) > 1e-10
            if np.count_nonzero(good_t) < 2:
                continue

            uv_keep = uv_keep[good_t]
            xyz_keep = xyz_keep[good_t]
            rho_keep = rho_keep[good_t]
            t_uv_keep = t_uv_keep[good_t]

            tree = cKDTree(uv_keep)
            if uv_keep.shape[0] > 1:
                nn_d, _ = tree.query(uv_keep, k=2)
                step_uv = float(np.median(nn_d[:, 1]))
            else:
                step_uv = 0.0
            if not np.isfinite(step_uv) or step_uv <= 1e-8:
                continue

            n_seed_face = min(
                max(4, int(round(max_paths / max(1, len(dense["per_face"]))))),
                uv_keep.shape[0],
            )
            seed_pick = np.linspace(0, uv_keep.shape[0] - 1, num=n_seed_face).round().astype(np.int64)
            covered = np.zeros(uv_keep.shape[0], dtype=bool)

            for seed_local in seed_pick.tolist():
                if covered[seed_local]:
                    continue

                back_uv, back_xyz = self._trace_streamline_on_dense_face(
                    uv_np=uv_keep,
                    xyz_np=xyz_keep,
                    t_uv_np=t_uv_keep,
                    rho_np=rho_keep,
                    start_idx=seed_local,
                    step_uv=step_uv,
                    max_steps=max_steps,
                    kdtree=tree,
                    align_thresh=min_align,
                    min_rho=thr,
                    sign=-1.0,
                )
                fwd_uv, fwd_xyz = self._trace_streamline_on_dense_face(
                    uv_np=uv_keep,
                    xyz_np=xyz_keep,
                    t_uv_np=t_uv_keep,
                    rho_np=rho_keep,
                    start_idx=seed_local,
                    step_uv=step_uv,
                    max_steps=max_steps,
                    kdtree=tree,
                    align_thresh=min_align,
                    min_rho=thr,
                    sign=1.0,
                )

                path_xyz = np.vstack([back_xyz[:0:-1], fwd_xyz])
                path_uv = np.vstack([back_uv[:0:-1], fwd_uv])
                if path_xyz.shape[0] < 3:
                    continue

                qd, qi = tree.query(path_uv, k=1)
                covered[qi[np.isfinite(qd)]] = True
                paths_xyz.append(path_xyz.astype(np.float32))
                path_rho.append(float(np.mean(rho_keep[qi[np.isfinite(qd)]])))

                if len(paths_xyz) >= int(max_paths):
                    break

            if len(paths_xyz) >= int(max_paths):
                break

        if len(paths_xyz) == 0:
            print(f"No valid fiber paths could be traced for threshold {thr:.3f}.")
            return {
                "points_xyz": points_np,
                "rho": density_np,
                "fiber3d": fiber.numpy(),
                "keep_mask": keep.numpy(),
                "paths_xyz": [],
            }

        line_points = []
        line_cells = []
        offset = 0
        for coords in paths_xyz:
            line_points.append(coords)
            line_cells.append(
                np.concatenate((
                    np.array([coords.shape[0]], dtype=np.int64),
                    np.arange(offset, offset + coords.shape[0], dtype=np.int64),
                ))
            )
            offset += coords.shape[0]

        line_points = np.vstack(line_points).astype(np.float32)
        line_cells = np.concatenate(line_cells).astype(np.int64)
        path_rho = np.asarray(path_rho, dtype=np.float32)

        plotter = pv.Plotter()
        surface = pv.PolyData(points_np, pv_faces_fixed)
        surface["rho"] = density_np
        plotter.add_mesh(
            surface,
            scalars="rho",
            cmap="Greys",
            opacity=0.20,
            show_edges=False,
        )

        lines = pv.PolyData()
        lines.points = line_points
        lines.lines = line_cells
        lines.cell_data["rho"] = path_rho
        plotter.add_mesh(lines, scalars="rho", cmap="plasma", line_width=float(line_width))
        plotter.show_axes()
        plotter.show()

        print(
            f"Fiber-path visualization: showing {len(paths_xyz)} paths "
            f"with threshold {thr:.3f}"
        )

        return {
            "paths_xyz": paths_xyz,
            "path_points": line_points,
            "path_rho": path_rho,
            "points_xyz": points_np,
            "rho": density_np,
            "fiber3d": fiber.numpy(),
            "keep_mask": keep.numpy(),
        }

    def visualize_result_final_fiber_paths(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Paths(*args, **kwargs)
    

    def visualize_result_final_smooth_surface_pyvista(
        self,
        result,
        shape_or_path=None,
        thr: float | str | None = 0.5,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        show_density: bool = True,
        auto_target_volfrac: float | None = None,
        dense_factor: float = 1.0,
    ):
        import pyvista as pv
        import numpy as np

        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )

        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
        )

        # ------------------------------------------------------------
        # Area-weighted global stats on dense CAD-native samples
        # ------------------------------------------------------------
        rho_all = []
        area_w_all = []
        for face_data in dense["per_face"]:
            rho_i = face_data["rho_dense"]
            Xu_i = face_data["Xu_dense"]
            Xv_i = face_data["Xv_dense"]
            area_w_i = torch.linalg.norm(torch.cross(Xu_i, Xv_i, dim=1), dim=1).clamp_min(self.cfg.eps)
            rho_all.append(rho_i.detach().cpu().numpy())
            area_w_all.append(area_w_i.detach().cpu().numpy())

        rho_all = np.concatenate(rho_all, axis=0)
        area_w_all = np.concatenate(area_w_all, axis=0)
        area_w_sum = float(area_w_all.sum()) + float(self.cfg.eps)
        volfrac_cont = float((rho_all * area_w_all).sum() / area_w_sum)

        thr_used = thr
        if thr is None or (isinstance(thr, str) and str(thr).lower() == "auto"):
            target = self.cfg.target_volfrac if auto_target_volfrac is None else float(auto_target_volfrac)
            target = float(np.clip(target, 0.0, 1.0))

            # Weighted quantile so that area fraction above threshold ~= target.
            q = 1.0 - target
            order = np.argsort(rho_all)
            rho_s = rho_all[order]
            w_s = area_w_all[order]
            cdf = np.cumsum(w_s) / (np.sum(w_s) + float(self.cfg.eps))
            thr_used = float(np.interp(q, cdf, rho_s))
        else:
            thr_used = float(thr)

        volfrac_thr = float(area_w_all[rho_all >= thr_used].sum() / area_w_sum)
        print(
            f"[smooth_surface] thr={thr_used:.4f} | "
            f"volfrac_cont(rho)={volfrac_cont:.4f} | "
            f"volfrac_thr(binary)={volfrac_thr:.4f} | "
            f"target={self.cfg.target_volfrac:.4f} | "
            f"grid=({grid_res_u} x {grid_res_v})"
        )

        plotter = pv.Plotter()

        for face_data in dense["per_face"]:
            uv_dense = face_data["uv_dense"]
            xyz = face_data["xyz_dense"].detach().cpu().numpy()
            rho = face_data["rho_dense"].detach().cpu().numpy()
            face_id = face_data["face_id"]
            ft = next(ft_i for ft_i in result["face_tensors"] if ft_i["face_id"] == face_id)
            u_periodic = bool(ft.get("u_periodic", False))
            v_periodic = bool(ft.get("v_periodic", False))

            Nu, Nv = face_data["grid_shape"]

            # We need full grid mapping
            mask = face_data["mask_dense_valid"].cpu().numpy()

            # Build full grid index map
            full_indices = -np.ones(mask.shape[0], dtype=int)
            full_indices[mask] = np.arange(mask.sum())

            faces = []

            def idx(i, j):
                return i * Nv + j

            max_i = Nu if u_periodic else (Nu - 1)
            max_j = Nv if v_periodic else (Nv - 1)

            for i in range(max_i):
                i2g = self._wrapped_grid_next(i, Nu, u_periodic)
                if i2g is None:
                    continue
                for j in range(max_j):
                    j2g = self._wrapped_grid_next(j, Nv, v_periodic)
                    if j2g is None:
                        continue
                    ids = [
                        idx(i, j),
                        idx(i, j2g),
                        idx(i2g, j),
                        idx(i2g, j2g),
                    ]

                    mapped = [full_indices[k] for k in ids]

                    if any(m < 0 for m in mapped):
                        continue

                    i0, i1, i2, i3 = mapped

                    # triangle 1
                    if rho[i0] >= thr_used and rho[i1] >= thr_used and rho[i2] >= thr_used:
                        faces.append([3, i0, i1, i2])

                    # triangle 2
                    if rho[i2] >= thr_used and rho[i1] >= thr_used and rho[i3] >= thr_used:
                        faces.append([3, i2, i1, i3])

            if len(faces) == 0:
                continue

            faces = np.array(faces).reshape(-1)

            mesh = pv.PolyData(xyz, faces)

            if show_density:
                mesh["rho"] = rho
                plotter.add_mesh(mesh, scalars="rho", cmap="viridis", clim=[0, 1])
            else:
                plotter.add_mesh(mesh, color="lightblue")
        plotter.show()
        return {
            "thr_used": float(thr_used),
            "volfrac_cont": float(volfrac_cont),
            "volfrac_thr": float(volfrac_thr),
            "dense": dense,
        }

    def visualize_boundary_attachment_debug(
        self,
        result: dict,
        shape_or_path=None,
        face_index: int = 0,
        grid_res_u: int = 180,
        grid_res_v: int = 180,
        uv_mask_tol: float | None = None,
        dense_factor: float = 1.0,
    ):
        """
        Debug boundary attachment on a single face:
        1) boundary samples in UV
        2) dmin to boundary samples
        3) decoder boundary field rho_b
        """
        import matplotlib.pyplot as plt

        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )

        face_tensors = result["face_tensors"]
        decoders = result["decoders"]
        pred_list = result["best_pred"]

        if not (0 <= face_index < len(face_tensors)):
            raise IndexError(f"face_index out of range: {face_index} not in [0, {len(face_tensors)-1}]")

        if shape_or_path is None:
            shape_or_path = result.get("shape_path", None)
        if shape_or_path is None:
            raise ValueError("shape_or_path is required (pass explicitly or store in result['shape_path']).")

        ft = face_tensors[face_index]
        decoder = decoders[face_index]
        face_id = ft["face_id"]

        pred_by_face_id = {p["face_id"]: p for p in pred_list}
        if face_id not in pred_by_face_id:
            raise KeyError(f"Missing best_pred for face_id={face_id}")
        pred = pred_by_face_id[face_id]

        sampled = self.sample_face_field_for_visualization(
            ft=ft,
            decoder=decoder,
            pred=pred,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_boundary_attachment=True,
        )

        uv_dense = sampled["uv_dense"]
        rho_b = sampled["rho_b_dense"]

        bidx = self._true_open_boundary_idx(ft)
        if bidx.numel() > 0:
            boundary_uv = ft["uv"][bidx]
            dmin = torch.cdist(uv_dense, boundary_uv).amin(dim=1)
        else:
            boundary_uv = None
            dmin = torch.full(
                (uv_dense.shape[0],),
                float("nan"),
                dtype=uv_dense.dtype,
                device=uv_dense.device,
            )

        uv_np = uv_dense.detach().cpu().numpy()
        rho_b_np = rho_b.detach().cpu().numpy()
        dmin_np = dmin.detach().cpu().numpy()
        bnd_np = None if boundary_uv is None else boundary_uv.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

        axes[0].scatter(uv_np[:, 0], uv_np[:, 1], s=2, c="lightgray", alpha=0.7)
        if bnd_np is not None and len(bnd_np) > 0:
            axes[0].scatter(bnd_np[:, 0], bnd_np[:, 1], s=10, c="red")
        axes[0].set_title(f"Face {face_id}: Boundary UV samples")
        axes[0].set_xlabel("u")
        axes[0].set_ylabel("v")
        axes[0].set_aspect("equal")

        sc1 = axes[1].scatter(uv_np[:, 0], uv_np[:, 1], c=dmin_np, s=3, cmap="viridis")
        axes[1].set_title("dmin to boundary")
        axes[1].set_xlabel("u")
        axes[1].set_ylabel("v")
        axes[1].set_aspect("equal")
        fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

        sc2 = axes[2].scatter(uv_np[:, 0], uv_np[:, 1], c=rho_b_np, s=3, cmap="magma", vmin=0.0, vmax=1.0)
        axes[2].set_title("rho_b (boundary field)")
        axes[2].set_xlabel("u")
        axes[2].set_ylabel("v")
        axes[2].set_aspect("equal")
        fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

        # Robust display across notebook/non-interactive backends.
        shown = False
        try:
            from IPython.display import display
            display(fig)
            shown = True
        except Exception:
            shown = False

        if not shown:
            import os
            os.makedirs("debug_plots", exist_ok=True)
            out_path = os.path.join("debug_plots", f"boundary_debug_face_{face_id}.png")
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            print(f"Saved debug figure to: {out_path}")

        plt.close(fig)

        finite_dmin = torch.isfinite(dmin)
        dmin_min = float(dmin[finite_dmin].min().item()) if bool(finite_dmin.any().item()) else float("nan")
        dmin_mean = float(dmin[finite_dmin].mean().item()) if bool(finite_dmin.any().item()) else float("nan")
        dmin_max = float(dmin[finite_dmin].max().item()) if bool(finite_dmin.any().item()) else float("nan")

        print(
            f"[Face {face_id}] "
            f"boundary_pts={0 if boundary_uv is None else int(boundary_uv.shape[0])} | "
            f"dmin(min/mean/max)=({dmin_min:.3e}/{dmin_mean:.3e}/{dmin_max:.3e}) | "
            f"rho_b(min/mean/max)=({float(rho_b.min().item()):.3e}/{float(rho_b.mean().item()):.3e}/{float(rho_b.max().item()):.3e}) | "
            f"grid=({grid_res_u} x {grid_res_v})"
        )

        return {
            "face_id": face_id,
            "uv_dense": uv_dense,
            "boundary_uv": boundary_uv,
            "dmin": dmin,
            "rho_b": rho_b,
            "sampled": sampled,
        }

    def report_threshold_sweep(
        self,
        result: dict,
        shape_or_path=None,
        thr_values=None,
        target_volfrac: float | None = None,
        volfrac_tol: float = 0.03,
        min_boundary_coverage: float = 0.95,
    ):
        """
        Sweep thresholds and report:
        - binary volume fraction (area-weighted)
        - boundary coverage (fraction of boundary samples classified solid)

        Selection rule:
        - among thresholds with boundary_coverage >= min_boundary_coverage and
          abs(volfrac_thr - target) <= volfrac_tol, pick the smallest threshold.
        - if none satisfies both, pick threshold with boundary coverage >= minimum
          and closest volume to target.
        - if still none, pick closest-volume threshold overall.
        """
        import numpy as np

        if thr_values is None:
            thr_values = np.linspace(0.05, 0.50, 19)
        thr_values = [float(t) for t in thr_values]

        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=300,
            grid_res_v=300,
            uv_mask_tol=None,
            use_best_pred=True,
        )

        face_tensors = result["face_tensors"]
        eps = float(self.cfg.eps)

        # global area-weighted continuous volume fraction
        rho_all = []
        area_all = []
        for face_data in dense["per_face"]:
            rho_i = face_data["rho_dense"]
            Xu_i = face_data["Xu_dense"]
            Xv_i = face_data["Xv_dense"]
            area_i = torch.linalg.norm(torch.cross(Xu_i, Xv_i, dim=1), dim=1).clamp_min(self.cfg.eps)
            rho_all.append(rho_i.detach().cpu().numpy())
            area_all.append(area_i.detach().cpu().numpy())
        rho_all = np.concatenate(rho_all, axis=0)
        area_all = np.concatenate(area_all, axis=0)
        area_sum = float(area_all.sum()) + eps
        volfrac_cont = float((rho_all * area_all).sum() / area_sum)

        target = self.cfg.target_volfrac if target_volfrac is None else float(target_volfrac)

        rows = []
        for thr in thr_values:
            # area-weighted binary volume fraction
            volfrac_thr = float(area_all[rho_all >= thr].sum() / area_sum)

            # boundary coverage: nearest dense point to each boundary sample
            bcov_faces = []
            for ft, face_data in zip(face_tensors, dense["per_face"]):
                bidx = self._true_open_boundary_idx(ft)
                if bidx.numel() == 0:
                    continue

                boundary_uv = ft["uv"][bidx]
                uv_dense = face_data["uv_dense"]
                rho_dense = face_data["rho_dense"]

                if uv_dense.numel() == 0:
                    continue

                d = torch.cdist(boundary_uv, uv_dense)
                nn = d.argmin(dim=1)
                cov = (rho_dense[nn] >= thr).float().mean()
                bcov_faces.append(float(cov.detach().item()))

            boundary_cov = float(np.mean(bcov_faces)) if len(bcov_faces) > 0 else float("nan")
            vol_err = abs(volfrac_thr - target)

            rows.append({
                "thr": float(thr),
                "volfrac_thr": volfrac_thr,
                "vol_err": float(vol_err),
                "boundary_cov": boundary_cov,
            })

        # Acceptable candidates
        acceptable = [
            r for r in rows
            if (not np.isnan(r["boundary_cov"]))
            and (r["boundary_cov"] >= min_boundary_coverage)
            and (r["vol_err"] <= volfrac_tol)
        ]

        if len(acceptable) > 0:
            best = min(acceptable, key=lambda r: r["thr"])
            reason = "smallest thr meeting boundary+volume criteria"
        else:
            boundary_ok = [
                r for r in rows
                if (not np.isnan(r["boundary_cov"])) and (r["boundary_cov"] >= min_boundary_coverage)
            ]
            if len(boundary_ok) > 0:
                best = min(boundary_ok, key=lambda r: (r["vol_err"], r["thr"]))
                reason = "closest volume among boundary-acceptable thresholds"
            else:
                best = min(rows, key=lambda r: (r["vol_err"], r["thr"]))
                reason = "closest volume overall (boundary criterion unmet)"

        print(
            f"[thr_sweep] target={target:.4f} | volfrac_cont={volfrac_cont:.4f} | "
            f"min_boundary_cov={min_boundary_coverage:.3f} | vol_tol={volfrac_tol:.3f}"
        )
        print("thr\tvolfrac_thr\tvol_err\tboundary_cov")
        for r in rows:
            bc = r["boundary_cov"]
            bc_s = "nan" if np.isnan(bc) else f"{bc:.4f}"
            print(f"{r['thr']:.4f}\t{r['volfrac_thr']:.4f}\t{r['vol_err']:.4f}\t{bc_s}")
        print(
            f"[thr_sweep] selected_thr={best['thr']:.4f} | "
            f"volfrac_thr={best['volfrac_thr']:.4f} | "
            f"boundary_cov={best['boundary_cov']:.4f} | reason={reason}"
        )

        return {
            "rows": rows,
            "selected": best,
            "reason": reason,
            "target_volfrac": float(target),
            "volfrac_cont": float(volfrac_cont),
            "min_boundary_coverage": float(min_boundary_coverage),
            "volfrac_tol": float(volfrac_tol),
        }

    def train(self, shape_path, face_tensors):
        cfg = self.cfg

        # validate the input shape tensors before training
        self._validate_face_tensors(face_tensors)

        # Assign device and data type used during training process
        ref_uv = face_tensors[0]["uv"]
        device = ref_uv.device
        dtype = ref_uv.dtype
        mid_step = cfg.num_steps // 2

        # Total number of points used for training
        all_global_idx = torch.cat([ft["global_vertex_idx"] for ft in face_tensors], dim=0)
        vertices_number = int(all_global_idx.max().item()) + 1

        Gating_binirize_sharpness = cfg.Gating_binirize_sharpness
        Gating_binirize_limit = cfg.Gating_binirize_limit

        # ------------------------------------------------------------
        # Build global vertex areas
        # ------------------------------------------------------------
        A_v = torch.zeros((vertices_number,), dtype=dtype, device=device)
        local_vertex_areas = []
        local_face_weights = []

        for ft in face_tensors:
            gidx = ft["global_vertex_idx"]
            A_local = self.generator.vertex_area_lumped(
                ft["uv"].shape[0],
                ft["faces_ijk"],
                ft["face_areas"],
            ).to(device=device, dtype=dtype)

            local_vertex_areas.append(A_local)
            local_face_weights.append(A_local.sum().clamp_min(cfg.eps))
            A_v[gidx] += A_local

        # ------------------------------------------------------------
        # Build models / optimizer / scheduler
        # ------------------------------------------------------------
        decoders, ppnets = self._build_face_models(face_tensors=face_tensors, device=device)

        uv_init_list = self._init_face_seeds(face_tensors)

        contexts = [
            torch.zeros(1, cfg.context_vector_size, device=device, dtype=dtype)
            for _ in face_tensors
        ]

        opt = self._build_optimizer(ppnets, decoders)

        raw_milestones = getattr(cfg, "scheduler_milestones", None)
        if raw_milestones is None:
            milestones = []
        else:
            raw_seq = [raw_milestones] if isinstance(raw_milestones, (int, float)) else list(raw_milestones)
            milestones = []
            for m in raw_seq:
                m = float(m)
                # Support both fractional milestones (0..1] and absolute step indices (>1).
                step_m = int(round(m * cfg.num_steps)) if m <= 1.0 else int(round(m))
                if 0 < step_m < cfg.num_steps:
                    milestones.append(step_m)
            milestones = sorted(set(milestones))

        print(f"scheduler_milestones: {milestones}")

        scheduler = None
        if getattr(cfg, "scheduler_milestones", None):
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=list(milestones),
                gamma=cfg.scheduler_gamma,
            )

        # ------------------------------------------------------------
        # Optional timelapse setup
        # ------------------------------------------------------------
        recorder = None
        render_cache = None
        if cfg.MakeTimelaps:
            case_name = shape_path.stem
            recorder = TimelapseRecorder(
                out_dir="timelapse_frames",
                video_path=case_name + "_timelapse.avi",
                fps=8,
            )
            render_cache = self.build_timelapse_render_cache(
                face_tensors=face_tensors,
                shape_or_path=str(shape_path),
                grid_res_u=cfg.TM_laps_res_u,
                grid_res_v=cfg.TM_laps_res_v,
                uv_mask_tol=None,
            )

        # ------------------------------------------------------------
        # Loss normalizers
        # ------------------------------------------------------------
        norm_vol = RunningNorm()
        norm_rep = RunningNorm()
        norm_bnd = RunningNorm()
        norm_strut = RunningNorm()
        norm_fem = RunningNorm()

        # ------------------------------------------------------------
        # Best-state tracking
        # ------------------------------------------------------------
        best_score = float("inf")
        best_vol_frac = None
        best_comp = None
        best_w_geo = None
        best_step = -1
        best_rho = None
        best_fiber_surface = None
        best_seeds = None
        best_pred = None
        steps_since_improve = 0

        initial_shape_density = None
        mid_shape_density = None
        final_shape_density = None
        final_shape_fiber_direction = None

        seed_points_init = None
        seed_points_mid = None
        seed_points_final = None

        rho0 = None
        seeds0 = None
        history = []

        self.current_face_tensors = face_tensors

        # ------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------
        with tqdm(
            range(cfg.num_steps),
            desc="Training",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for step in pbar:
                use_hard_gate_mask_step = self._use_hard_gate_mask(
                    step=step,
                    steps_since_improve=steps_since_improve,
                )
                rho_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
                rho_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                rho_b_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
                rho_b_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                rho_v_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
                rho_v_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                rho_s_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
                rho_s_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                fiber_acc = torch.zeros((vertices_number, 3), dtype=dtype, device=device)
                fiber_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                seeds_list = []
                pred_list = []

                rep_terms = []
                bnd_terms = []
                strut_terms = []
                strut_edge_terms = []
                strut_void_terms = []
                w_geo_terms = []
                h_terms = []

                boundary_width_terms = []
                boundary_alpha_terms = []
                boundary_beta_terms = []

                theta_mean_terms = []
                a_metric_terms = []

                gate_terms = []
                gate_binary_terms = []
                face_weights_this_step = []

                active_count_total = 0.0
                active_frac_sum = 0.0
                gate_raw_min_list = []
                gate_raw_mean_sum = 0.0
                gate_raw_max_list = []
                gate_sharp_min_list = []
                gate_sharp_mean_sum = 0.0
                gate_sharp_max_list = []
                gate_decoder_min_list = []
                gate_decoder_mean_sum = 0.0
                gate_decoder_max_list = []
                gate_face_count = 0

                # ----------------------------------------------------
                # Per-face forward pass
                # ----------------------------------------------------
                for ft, decoder, ppnet, uv_init_i, context_i, A_local, face_weight_i in zip(
                    face_tensors,
                    decoders,
                    ppnets,
                    uv_init_list,
                    contexts,
                    local_vertex_areas,
                    local_face_weights,
                ):
                    pred_i = ppnet(context_i, uv_init_i, offset_scale=cfg.Offset_scale)

                    seeds_raw_i = pred_i["seeds_raw"][0]
                    w_raw_i = pred_i["w_raw"][0]

                    h_raw_i = None
                    if cfg.fixed_height is None and "h_raw" in pred_i:
                        h_raw_i = pred_i["h_raw"][0]

                    gate_logits_i = pred_i.get("gate_logits", None)
                    gate_logits_i = gate_logits_i[0] if gate_logits_i is not None else None

                    gate_probs_raw_i = pred_i.get("gate_probs", None)
                    gate_probs_raw_i = gate_probs_raw_i[0] if gate_probs_raw_i is not None else None

                    gates_struct_i = self.sharpen_gate_probs(
                        gate_probs_raw_i,
                        Gating_binirize_sharpness=Gating_binirize_sharpness,
                        Gating_binirize_limit = Gating_binirize_limit,
                        eps=cfg.gate_eps,
                    )
                    gates_decoder_i = self._decoder_gate_values(
                        gates_struct=gates_struct_i,
                        use_hard_gate_mask=use_hard_gate_mask_step,
                        threshold=Gating_binirize_limit,
                    )

                    theta_i = pred_i["theta"][0] if (cfg.use_Metric_anisotropy and "theta" in pred_i) else None
                    a_raw_i = pred_i["a_raw"][0] if (cfg.use_Metric_anisotropy and "a_raw" in pred_i) else None

                    boundary_width_raw_i = pred_i["boundary_width_raw"][0] if ("boundary_width_raw" in pred_i) else None
                    boundary_alpha_raw_i = None
                    boundary_beta_raw_i = None

                    if gates_decoder_i is not None:
                        gate_stats_i = self._gate_activity_stats(
                            gate_probs=gates_decoder_i,
                            threshold=Gating_binirize_limit,
                        )
                        gate_raw_stats_i = self._gate_stats(gate_probs_raw_i)
                        gate_sharp_stats_i = self._gate_stats(gates_struct_i)
                        gate_decoder_stats_i = self._gate_stats(gates_decoder_i)
                        active_count_total += gate_stats_i["active_count"]
                        active_frac_sum += gate_stats_i["active_frac"]
                        gate_raw_min_list.append(gate_raw_stats_i["min"])
                        gate_raw_mean_sum += gate_raw_stats_i["mean"]
                        gate_raw_max_list.append(gate_raw_stats_i["max"])
                        gate_sharp_min_list.append(gate_sharp_stats_i["min"])
                        gate_sharp_mean_sum += gate_sharp_stats_i["mean"]
                        gate_sharp_max_list.append(gate_sharp_stats_i["max"])
                        gate_decoder_min_list.append(gate_decoder_stats_i["min"])
                        gate_decoder_mean_sum += gate_decoder_stats_i["mean"]
                        gate_decoder_max_list.append(gate_decoder_stats_i["max"])
                        gate_face_count += 1

                        #gate_count_loss_i = torch.zeros((), dtype=gate_probs_raw_i.dtype, device=gate_probs_raw_i.device)
                        gate_count_loss_i = self.gate_count_loss(
                            gates=gate_probs_raw_i,
                            target_count=cfg.gate_target_count,
                        )
                        gate_terms.append(gate_count_loss_i)

                        gate_binary_loss_i = self.gate_binary_loss(gates_decoder_i)
                        gate_binary_terms.append(gate_binary_loss_i)



                    local_face_id = torch.zeros(ft["uv"].shape[0], dtype=torch.long, device=device)

                    boundary_uv_i = None
                    boundary_face_id_i = None
                    true_bidx_i = self._true_open_boundary_idx(ft)
                    if true_bidx_i.numel() > 0:
                        boundary_uv_i = ft["uv"][true_bidx_i]
                        boundary_face_id_i = torch.zeros(
                            boundary_uv_i.shape[0],
                            dtype=torch.long,
                            device=device,
                        )

                    decoder_out = decoder(
                        points_uv=ft["uv"],
                        Xu=ft["Xu"],
                        Xv=ft["Xv"],
                        tau=cfg.tau,
                        seeds_raw=seeds_raw_i,
                        w_raw=w_raw_i,
                        h_raw=h_raw_i,
                        theta=theta_i,
                        a_raw=a_raw_i,
                        points_face_id=local_face_id,
                        boundary_uv=boundary_uv_i,
                        boundary_face_id=boundary_face_id_i,
                        boundary_width_raw=boundary_width_raw_i,
                        boundary_alpha_raw=boundary_alpha_raw_i,
                        boundary_beta_raw=boundary_beta_raw_i,
                        seed_gates=gates_decoder_i,
                    )

                    self._require_decoder_keys(
                        decoder_out,
                        [
                            "seeds",
                            "rho",
                            "rho_s",
                            "fiber3d",
                            "w_geo",
                            "rho_v",
                            "rho_b",
                            "edge_field",
                            "boundary_width",
                            "boundary_alpha",
                            "boundary_beta",
                        ],
                    )

                    seeds_i = decoder_out["seeds"]
                    rho_i = decoder_out["rho"]
                    rho_s_i = decoder_out["rho_s"]
                    w_geo_i = decoder_out["w_geo"]
                    fiber3d_i = decoder_out["fiber3d"]
                    rho_v_i = decoder_out["rho_v"]
                    rho_b_i = decoder_out["rho_b"]
                    edge_field_i = decoder_out["edge_field"]

                    boundary_width_i = decoder_out["boundary_width"]
                    boundary_alpha_i = decoder_out["boundary_alpha"]
                    boundary_beta_i = decoder_out["boundary_beta"]

                    h_i = decoder_out["h"]

                    for name, t in {
                        "seeds_i": seeds_i,
                        "rho_i": rho_i,
                        "rho_s_i": rho_s_i,
                        "fiber3d_i": fiber3d_i,
                        "rho_v_i": rho_v_i,
                        "rho_b_i": rho_b_i,
                        "edge_field_i": edge_field_i,
                    }.items():
                        if not torch.isfinite(t).all():
                            tqdm.write(f"[step {step}] face {ft['face_id']} invalid tensor: {name}")
                            raise RuntimeError(
                                f"Invalid decoder output on face {ft['face_id']} at step {step}"
                            )

                    gidx = ft["global_vertex_idx"]
                    w_local = A_local.clamp_min(cfg.eps)

                    rho_acc[gidx] += rho_i * w_local
                    rho_wgt[gidx] += w_local

                    rho_b_acc[gidx] += rho_b_i * w_local
                    rho_b_wgt[gidx] += w_local

                    rho_v_acc[gidx] += rho_v_i * w_local
                    rho_v_wgt[gidx] += w_local

                    rho_s_acc[gidx] += rho_s_i * w_local
                    rho_s_wgt[gidx] += w_local

                    fiber_acc[gidx] += fiber3d_i * w_local[:, None]
                    fiber_wgt[gidx] += w_local

                    seeds_list.append(seeds_i)

                    pred_list.append({
                        "face_id": ft["face_id"],
                        "seeds_raw": seeds_raw_i.detach().clone(),
                        "w_raw": w_raw_i.detach().clone(),
                        "h_raw": None if h_raw_i is None else h_raw_i.detach().clone(),
                        "gate_logits": None if gate_logits_i is None else gate_logits_i.detach().clone(),
                        "gate_probs_raw": None if gate_probs_raw_i is None else gate_probs_raw_i.detach().clone(),
                        "gate_probs_sharp": None if gates_struct_i is None else gates_struct_i.detach().clone(),
                        "gate_probs_decoder": None if gates_decoder_i is None else gates_decoder_i.detach().clone(),
                        "gate_probs": None if gates_decoder_i is None else gates_decoder_i.detach().clone(),
                        "theta": None if theta_i is None else theta_i.detach().clone(),
                        "a_raw": None if a_raw_i is None else a_raw_i.detach().clone(),

                        "boundary_width": boundary_width_i.detach().clone() if isinstance(boundary_width_i, torch.Tensor) else boundary_width_i,
                        "boundary_alpha": boundary_alpha_i.detach().clone() if isinstance(boundary_alpha_i, torch.Tensor) else boundary_alpha_i,
                        "boundary_beta": boundary_beta_i.detach().clone() if isinstance(boundary_beta_i, torch.Tensor) else boundary_beta_i,

                        "w_geo": w_geo_i.detach().clone(),
                        "h": h_i.detach().clone() if isinstance(h_i, torch.Tensor) else h_i,

                        "boundary_width_raw": None if boundary_width_raw_i is None else boundary_width_raw_i.detach().clone(),
                        "boundary_alpha_raw": None,
                        "boundary_beta_raw": None,

                        "theta_mean": None if theta_i is None else theta_i.mean().detach().clone(),
                        "a_metric": None if a_raw_i is None else (
                            0.5 * (2.0 - 0.5) * torch.tanh(a_raw_i) + 0.5 * (2.0 + 0.5)
                        ).mean().detach().clone(),
                    })

                    rep_terms.append(
                        self.seed_repulsion_term(
                            seeds=seeds_i,
                            gates=gates_decoder_i,
                            sigma=cfg.seed_repulsion_sigma,
                            eps=cfg.eps,
                        )
                    )

                    bnd_terms.append(
                        self.boundary_repulsion_term(
                            seeds=seeds_i,
                            boundary_uv=boundary_uv_i,
                            gates=gates_decoder_i,
                            margin=cfg.boundary_margin,
                            eps=cfg.eps,
                        )
                    )

                    w_geo_terms.append(self._pair_upper_values(w_geo_i).mean().reshape(()))
                    h_terms.append(h_i.reshape(()))

                    if isinstance(boundary_width_i, torch.Tensor) and boundary_width_i.numel() > 0:
                        boundary_width_terms.append(boundary_width_i.reshape(()))
                    if isinstance(boundary_alpha_i, torch.Tensor) and boundary_alpha_i.numel() > 0:
                        boundary_alpha_terms.append(boundary_alpha_i.reshape(()))
                    if isinstance(boundary_beta_i, torch.Tensor) and boundary_beta_i.numel() > 0:
                        boundary_beta_terms.append(boundary_beta_i.reshape(()))

                    if theta_i is not None:
                        theta_mean_terms.append(theta_i.mean().reshape(()))
                    if a_raw_i is not None:
                        a_metric_i = 0.5 * (2.0 - 0.5) * torch.tanh(a_raw_i) + 0.5 * (2.0 + 0.5)
                        a_metric_terms.append(a_metric_i.mean().reshape(()))

                    face_weights_this_step.append(face_weight_i.reshape(()))

                    if cfg.lam_strut != 0.0:
                        loss_strut_i, loss_strut_edge_i, loss_strut_void_i = self.strutness_loss_from_edge_field(
                            rho=rho_v_i,
                            edge_field=edge_field_i,
                            lam_edge=cfg.lam_strut_edge,
                            lam_void=cfg.lam_strut_void,
                            eps=cfg.eps,
                        )
                        strut_terms.append(loss_strut_i)
                        strut_edge_terms.append(loss_strut_edge_i)
                        strut_void_terms.append(loss_strut_void_i)

                # ----------------------------------------------------
                # Aggregate face outputs
                # ----------------------------------------------------
                if gate_face_count > 0:
                    active_count_mean = active_count_total / gate_face_count
                    active_frac_mean = active_frac_sum / gate_face_count
                    gate_raw_min_global = min(gate_raw_min_list)
                    gate_raw_mean_global = gate_raw_mean_sum / gate_face_count
                    gate_raw_max_global = max(gate_raw_max_list)
                    gate_sharp_min_global = min(gate_sharp_min_list)
                    gate_sharp_mean_global = gate_sharp_mean_sum / gate_face_count
                    gate_sharp_max_global = max(gate_sharp_max_list)
                    gate_decoder_min_global = min(gate_decoder_min_list)
                    gate_decoder_mean_global = gate_decoder_mean_sum / gate_face_count
                    gate_decoder_max_global = max(gate_decoder_max_list)
                else:
                    active_count_mean = 0.0
                    active_frac_mean = 0.0
                    gate_raw_min_global = 0.0
                    gate_raw_mean_global = 0.0
                    gate_raw_max_global = 0.0
                    gate_sharp_min_global = 0.0
                    gate_sharp_mean_global = 0.0
                    gate_sharp_max_global = 0.0
                    gate_decoder_min_global = 0.0
                    gate_decoder_mean_global = 0.0
                    gate_decoder_max_global = 0.0

                rho = rho_acc / rho_wgt.clamp_min(cfg.eps)
                rho_boundary = rho_b_acc / rho_b_wgt.clamp_min(cfg.eps)
                rho_v_all = rho_v_acc / rho_v_wgt.clamp_min(cfg.eps)
                rho_s_all = rho_s_acc / rho_s_wgt.clamp_min(cfg.eps)

                fiber_surface = fiber_acc / fiber_wgt.clamp_min(cfg.eps)[:, None]
                fiber_norm = fiber_surface.norm(dim=1, keepdim=True).clamp_min(cfg.eps)
                fiber_surface = fiber_surface / fiber_norm

                loss_rep = self._safe_weighted_mean(rep_terms, face_weights_this_step, dtype, device, cfg.eps)
                loss_bnd = self._safe_weighted_mean(bnd_terms, face_weights_this_step, dtype, device, cfg.eps)
                loss_strut = self._safe_weighted_mean(strut_terms, face_weights_this_step, dtype, device, cfg.eps)
                loss_strut_edge = self._safe_weighted_mean(strut_edge_terms, face_weights_this_step, dtype, device, cfg.eps)
                loss_strut_void = self._safe_weighted_mean(strut_void_terms, face_weights_this_step, dtype, device, cfg.eps)

                w_geo_mean = self._safe_weighted_mean(w_geo_terms, face_weights_this_step, dtype, device, cfg.eps)
                h_mean = self._safe_weighted_mean(h_terms, face_weights_this_step, dtype, device, cfg.eps)

                boundary_width_mean = self._safe_weighted_mean(boundary_width_terms, face_weights_this_step, dtype, device, cfg.eps)
                boundary_alpha_mean = self._safe_weighted_mean(boundary_alpha_terms, face_weights_this_step, dtype, device, cfg.eps)
                boundary_beta_mean = self._safe_weighted_mean(boundary_beta_terms, face_weights_this_step, dtype, device, cfg.eps)

                theta_mean = self._safe_weighted_mean(theta_mean_terms, face_weights_this_step, dtype, device, cfg.eps)
                a_metric_mean = self._safe_weighted_mean(a_metric_terms, face_weights_this_step, dtype, device, cfg.eps)

                loss_gate = self._safe_weighted_mean(gate_terms, face_weights_this_step, dtype, device, cfg.eps)
                loss_gate_binary = self._safe_weighted_mean(gate_binary_terms, face_weights_this_step, dtype, device, cfg.eps)

                # ----------------------------------------------------
                # Volume loss
                # ----------------------------------------------------
                vol_frac_total = (rho * A_v).sum() / (A_v.sum() + cfg.eps)
                vol_frac_v = (rho_v_all * A_v).sum() / (A_v.sum() + cfg.eps)
                loss_vol_v = self.volume_loss_constant_height(
                    rho=rho_v_all,
                    A_v=A_v,
                    target_volfrac=cfg.target_volfrac,
                    eps=cfg.eps,
                )
                loss_vol_total = self.volume_loss_constant_height(
                    rho=rho,
                    A_v=A_v,
                    target_volfrac=cfg.target_volfrac,
                    eps=cfg.eps,
                )

                loss_vol_eff_v, vol_frac_eff = self.volume_loss_powered(
                    rho=rho_v_all,
                    A_v=A_v,
                    target_volfrac=cfg.target_volfrac,
                    power=cfg.effective_volume_power,
                    eps=cfg.eps,
                )
                sharp_vol_ramp = self.ramp_weight(
                    step=step,
                    total_steps=cfg.num_steps,
                    start_frac=cfg.sharp_vol_start_frac,
                    ramp_frac=cfg.sharp_vol_ramp_frac,
                )
                loss_vol_sharp, vol_frac_sharp = self.volume_loss_constant_height(
                    rho=rho_s_all,
                    A_v=A_v,
                    target_volfrac=cfg.target_volfrac,
                    eps=cfg.eps,
                ), (rho_s_all * A_v).sum() / (A_v.sum() + cfg.eps)

                loss_vol = (
                    loss_vol_v
                    + cfg.boundary_volume_assist * loss_vol_total
                    + cfg.lam_vol_effective * loss_vol_eff_v
                    + (cfg.lam_vol_sharp * sharp_vol_ramp) * loss_vol_sharp
                )

                if cfg.use_boundary_weighted_volume:
                    loss_vol_base, vol_frac_weighted = self.volume_loss_with_boundary_discount(
                        rho=rho,
                        A_v=A_v,
                        rho_boundary=rho_boundary,
                        target_volfrac=cfg.target_volfrac,
                        boundary_weight=cfg.boundary_vol_weight,
                        eps=cfg.eps,
                    )
                    loss_vol_eff_weighted, vol_frac_eff = self.volume_loss_powered(
                        rho=rho,
                        A_v=(1.0 - rho_boundary + cfg.boundary_vol_weight * rho_boundary) * A_v,
                        target_volfrac=cfg.target_volfrac,
                        power=cfg.effective_volume_power,
                        eps=cfg.eps,
                    )
                    sharp_weights = (1.0 - rho_boundary + cfg.boundary_vol_weight * rho_boundary) * A_v
                    loss_vol_sharp, vol_frac_sharp = self.volume_loss_constant_height(
                        rho=rho_s_all,
                        A_v=sharp_weights,
                        target_volfrac=cfg.target_volfrac,
                        eps=cfg.eps,
                    ), (rho_s_all * sharp_weights).sum() / (sharp_weights.sum() + cfg.eps)
                    loss_vol = (
                        loss_vol_base
                        + cfg.lam_vol_effective * loss_vol_eff_weighted
                        + (cfg.lam_vol_sharp * sharp_vol_ramp) * loss_vol_sharp
                    )
                    vol_frac_weighted_cont = vol_frac_weighted
                else:
                    vol_frac_weighted_cont = vol_frac_v



                # ----------------------------------------------------
                # FEM loss
                # ----------------------------------------------------
                fem_out = {
                    "fem_total": torch.zeros((), dtype=dtype, device=device),
                    "comp": torch.zeros((), dtype=dtype, device=device),
                    "compliance_loss": torch.zeros((), dtype=dtype, device=device),
                    "fem_valid": True,
                    "failure_reason": None,
                }

                if cfg.lam_fem != 0.0:
                    fem_out = self.fem_loss(
                        rho_surface=rho,
                        fiber_surface=fiber_surface,
                        comp_normalize_by=cfg.comp_normalize_by,
                        density_floor=cfg.fem_density_floor,
                        eps=cfg.eps,
                        save_debug_history=getattr(cfg, "save_fem_debug_history", True),
                    )

                loss_fem = fem_out["fem_total"]
                loss_comp = fem_out["compliance_loss"]
                comp_val = fem_out["comp"]
                fem_is_valid = bool(fem_out["fem_valid"])
                fem_failure_reason = fem_out["failure_reason"]

                # ----------------------------------------------------
                # Normalize losses
                # ----------------------------------------------------
                if cfg.normalize_losses:
                    n_vol = norm_vol.update(loss_vol.detach().item())
                    n_rep = norm_rep.update(loss_rep.detach().item())
                    n_bnd = norm_bnd.update(loss_bnd.detach().item())
                    n_strut = norm_strut.update(loss_strut.detach().item()) if cfg.lam_strut != 0.0 else 1.0
                    n_fem = norm_fem.update(loss_fem.detach().item()) if (cfg.lam_fem != 0.0 and fem_is_valid) else 1.0
                else:
                    n_vol = n_rep = n_bnd = n_strut = n_fem = 1.0

                # ----------------------------------------------------
                # Total loss
                # ----------------------------------------------------
                if cfg.gate_warmup_steps > 0:
                    gate_warmup = min(float(step) / float(cfg.gate_warmup_steps), 1.0)
                else:
                    gate_warmup = 1.0

                if cfg.gate_binary_warmup_steps > 0:
                    gate_binary_warmup = min(float(step) / float(cfg.gate_binary_warmup_steps), 1.0)
                else:
                    gate_binary_warmup = 1.0

                lam_gate_binary_eff = cfg.lam_gate_binary * gate_binary_warmup
                lam_gate_eff = cfg.lam_gate_count * gate_warmup

                L_total = (
                    cfg.lam_vol * (loss_vol / n_vol)
                    + cfg.lam_rep * (loss_rep / n_rep)
                    + cfg.lam_bnd * (loss_bnd / n_bnd)
                )

                if cfg.lam_strut != 0.0:
                    L_total = L_total + cfg.lam_strut * (loss_strut / n_strut)

                if cfg.use_gating and lam_gate_eff != 0.0:
                    L_total = L_total + lam_gate_eff * loss_gate

                if cfg.use_gating and lam_gate_binary_eff != 0.0:
                    L_total = L_total + lam_gate_binary_eff * loss_gate_binary

                if cfg.lam_fem != 0.0:
                    if fem_is_valid:
                        L_total = L_total + cfg.lam_fem * (loss_fem / n_fem)
                    elif not cfg.skip_bad_fem_steps:
                        L_total = L_total + cfg.lam_fem * loss_fem

                L_total = L_total / len(face_tensors)
                total_is_finite = self._scalar_tensor_is_finite(L_total)

                # ----------------------------------------------------
                # Backprop
                # ----------------------------------------------------
                opt.zero_grad(set_to_none=True)

                if total_is_finite:
                    L_total.backward()

                    grad_clip_norm = getattr(cfg, "grad_clip_norm", None)
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        params = []
                        for ppnet in ppnets:
                            params.extend([p for p in ppnet.parameters() if p.requires_grad])
                        if params:
                            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)

                    opt.step()

                    for fi, ppnet in enumerate(ppnets):
                        for pn, p in ppnet.named_parameters():
                            if not torch.isfinite(p).all():
                                raise RuntimeError(
                                    f"Non-finite parameter after opt.step(): face={fi}, param={pn}"
                                )

                    if scheduler is not None:
                        scheduler.step()
                else:
                    tqdm.write(f"[step {step}] L_total is non-finite, optimizer step skipped.")

                # ----------------------------------------------------
                # Logging / tracking
                # ----------------------------------------------------
                with torch.no_grad():
                    vol_frac = (rho * A_v).sum() / (A_v.sum() + cfg.eps)
                    vol_dev = torch.abs(vol_frac - cfg.target_volfrac)
                    vol_dev_eff = torch.abs(vol_frac_eff - cfg.target_volfrac)

                    score = float(L_total.detach().item()) if total_is_finite else float("inf")
                    best_candidate_is_valid = (cfg.lam_fem == 0.0) or fem_is_valid

                    prev_best_step = best_step
                    improvement_gap = (step - prev_best_step) if prev_best_step >= 0 else None

                    if step == 0:
                        initial_shape_density = rho.detach().clone()
                        seed_points_init = self._seed_points_xyz_all_faces(seeds_list, face_tensors)

                    if step == mid_step:
                        mid_shape_density = rho.detach().clone()
                        seed_points_mid = self._seed_points_xyz_all_faces(seeds_list, face_tensors)

                    if best_candidate_is_valid and score < (best_score - cfg.min_delta):
                        best_score = score
                        best_step = step
                        best_vol_frac = float(vol_frac_eff.detach().item())
                        best_comp = float(comp_val.detach().item())
                        best_w_geo = float(w_geo_mean.detach().item())
                        best_rho = rho.detach().clone()
                        best_fiber_surface = fiber_surface.detach().clone()
                        best_seeds = [s.detach().clone() for s in seeds_list]
                        best_pred = [
                            {
                                "face_id": p["face_id"],
                                "seeds_raw": p["seeds_raw"].detach().clone(),
                                "w_raw": p["w_raw"].detach().clone(),
                                "h_raw": None if p["h_raw"] is None else p["h_raw"].detach().clone(),
                                "gate_logits": None if p.get("gate_logits") is None else p["gate_logits"].detach().clone(),
                                "gate_probs_raw": None if p.get("gate_probs_raw") is None else p["gate_probs_raw"].detach().clone(),
                                "gate_probs_sharp": None if p.get("gate_probs_sharp") is None else p["gate_probs_sharp"].detach().clone(),
                                "gate_probs_decoder": None if p.get("gate_probs_decoder") is None else p["gate_probs_decoder"].detach().clone(),
                                "gate_probs": None if p["gate_probs"] is None else p["gate_probs"].detach().clone(),
                                "theta": None if p["theta"] is None else p["theta"].detach().clone(),
                                "a_raw": None if p["a_raw"] is None else p["a_raw"].detach().clone(),
                                "boundary_width_raw": None if p["boundary_width_raw"] is None else p["boundary_width_raw"].detach().clone(),
                                "boundary_alpha_raw": None if p["boundary_alpha_raw"] is None else p["boundary_alpha_raw"].detach().clone(),
                                "boundary_beta_raw": None if p["boundary_beta_raw"] is None else p["boundary_beta_raw"].detach().clone(),
                                "w_geo": p["w_geo"].detach().clone(),
                                "h": None if p["h"] is None else p["h"].detach().clone(),
                                "boundary_width": None if p["boundary_width"] is None else p["boundary_width"].detach().clone(),
                                "boundary_alpha": None if p["boundary_alpha"] is None else p["boundary_alpha"].detach().clone(),
                                "boundary_beta": None if p["boundary_beta"] is None else p["boundary_beta"].detach().clone(),
                                "theta_mean": None if p["theta_mean"] is None else p["theta_mean"].detach().clone(),
                                "a_metric": None if p["a_metric"] is None else p["a_metric"].detach().clone(),
                            }
                            for p in pred_list
                        ]

                        if improvement_gap is None or improvement_gap > 50:
                            tqdm.write(
                                f"New best_step={best_step} | "
                                f"best_score={best_score:.6f} | "
                                f"vol_eff={best_vol_frac:.6f} | "
                                f"comp={best_comp:.6e} | "
                                f"w={best_w_geo:.6e}"
                            )
                        steps_since_improve = 0
                    elif best_candidate_is_valid:
                        steps_since_improve += 1

                    if rho0 is None:
                        rho0 = rho.detach().clone()
                    if seeds0 is None:
                        seeds0 = [s.detach().clone() for s in seeds_list]

                    drho = float((rho - rho0).abs().mean().item())
                    dseed_terms = [float((s - s0).abs().mean().item()) for s, s0 in zip(seeds_list, seeds0)]
                    dseed = sum(dseed_terms) / max(len(dseed_terms), 1)

                    rho_min = float(rho.min().item())
                    rho_mean = float(rho.mean().item())
                    rho_max = float(rho.max().item())

                    rho_boundary_min = float(rho_boundary.min().item())
                    rho_boundary_mean = float(rho_boundary.mean().item())
                    rho_boundary_max = float(rho_boundary.max().item())

                    rho_v_min = float(rho_v_all.min().item())
                    rho_v_mean = float(rho_v_all.mean().item())
                    rho_v_max = float(rho_v_all.max().item())


                    g_mean = 0.0
                    g_count = 0
                    for ppnet in ppnets:
                        for p in ppnet.parameters():
                            if p.grad is not None:
                                g_mean += float(p.grad.detach().abs().mean().item())
                                g_count += 1
                    g_mean = g_mean / max(g_count, 1)

                    row = {
                        "step": step,
                        "L_total": self._finite_or_default(L_total),
                        "loss_vol": self._finite_or_default(loss_vol),
                        "loss_rep": self._finite_or_default(loss_rep),
                        "loss_bnd": self._finite_or_default(loss_bnd),
                        "loss_strut": self._finite_or_default(loss_strut),
                        "loss_strut_edge": self._finite_or_default(loss_strut_edge),
                        "loss_strut_void": self._finite_or_default(loss_strut_void),
                        "loss_fem": self._finite_or_default(loss_fem),
                        "loss_comp": self._finite_or_default(loss_comp),
                        "loss_gate": self._finite_or_default(loss_gate),
                        "lam_gate_eff": lam_gate_eff,
                        "loss_gate_binary": self._finite_or_default(loss_gate_binary),
                        "lam_gate_binary_eff": lam_gate_binary_eff,
                        "comp": self._finite_or_default(comp_val),
                        "vol_frac": float(vol_frac.detach().item()),
                        "vol_frac_eff": float(vol_frac_eff.detach().item()),
                        "vol_frac_sharp": float(vol_frac_sharp.detach().item()),
                        "vol_dev": float(vol_dev.detach().item()),
                        "vol_dev_eff": float(vol_dev_eff.detach().item()),
                        "rho_min": rho_min,
                        "rho_mean": rho_mean,
                        "rho_max": rho_max,
                        "rho_boundary_min": rho_boundary_min,
                        "rho_boundary_mean": rho_boundary_mean,
                        "rho_boundary_max": rho_boundary_max,
                        "rho_v_min": rho_v_min,
                        "rho_v_max": rho_v_max,
                        "rho_v_mean": float(rho_v_all.mean().detach().item()),
                        "drho": drho,
                        "dseed": dseed,
                        "grad_mean": g_mean,
                        "best_score": best_score,
                        "best_step": best_step,
                        "fem_valid": fem_is_valid,
                        "fem_failure_reason": fem_failure_reason,
                        "optimizer_step_skipped": not total_is_finite,
                        "loss_vol_sharp": self._finite_or_default(loss_vol_sharp),
                        "sharp_vol_ramp": float(sharp_vol_ramp),

                        "w_geo_mean": self._finite_or_default(w_geo_mean),
                        "h_mean": self._finite_or_default(h_mean),

                        "boundary_width_mean": self._finite_or_default(boundary_width_mean),
                        "boundary_alpha_mean": self._finite_or_default(boundary_alpha_mean),
                        "boundary_beta_mean": self._finite_or_default(boundary_beta_mean),

                        "theta_mean": self._finite_or_default(theta_mean),
                        "a_metric_mean": self._finite_or_default(a_metric_mean),

                        "active_count_total": active_count_total,
                        "active_count_mean": active_count_mean,
                        "active_frac_mean": active_frac_mean,
                        "gate_raw_min": gate_raw_min_global,
                        "gate_raw_mean": gate_raw_mean_global,
                        "gate_raw_max": gate_raw_max_global,
                        "gate_sharp_min": gate_sharp_min_global,
                        "gate_sharp_mean": gate_sharp_mean_global,
                        "gate_sharp_max": gate_sharp_max_global,
                        "gate_decoder_min": gate_decoder_min_global,
                        "gate_decoder_mean": gate_decoder_mean_global,
                        "gate_decoder_max": gate_decoder_max_global,
                        "hard_gate_mask_on": 1.0 if use_hard_gate_mask_step else 0.0,
                    }
                    history.append(row)

                    pbar.set_postfix(
                        loss=f"{row['L_total']:.3e}",
                        vol=f"{row['vol_frac_eff']:.3f}",
                        comp=f"{row['comp']:.2e}",
                        w=f"{row['w_geo_mean']:.3e}",
                        bw=f"{row['boundary_width_mean']:.3e}",
                        act=f"{active_count_mean:.1f}",
                        fem="OK" if fem_is_valid else "BAD",
                        refresh=False,
                    )

                    if cfg.MakeTimelaps and step % cfg.timelapse_frame_step == 0:
                        cad_img = self._render_current_cad_frame_cached(
                            seeds_list=seeds_list,
                            decoders=decoders,
                            pred_list=pred_list,
                            render_cache=render_cache,
                            thr=getattr(cfg, "vis_thr", cfg.TM_laps_Thr),
                        )

                        loss_dict = {
                            "total": row["L_total"],
                            "vol": row["loss_vol"],
                            "rep": row["loss_rep"],
                            "bnd": row["loss_bnd"],
                            "strut": row["loss_strut"],
                            "fem": row["loss_fem"],
                        }

                        recorder.add_frame(
                            step=step,
                            cad_img=cad_img,
                            loss_dict=loss_dict,
                            title_text=(
                                f"vol={row['vol_frac']:.4f} | "
                                f"W={row['w_geo_mean']:.4g} | "
                                f"bw={row['boundary_width_mean']:.4g} | "
                                f"Δrho={drho:.2e} Δseed={dseed:.2e} grad_mean={g_mean:.2e} | "
                            ),
                        )

                    self._tb_log_step(
                        step=step,
                        row=row,
                        rho=rho,
                        rho_boundary=rho_boundary,
                        rho_v_all=rho_v_all,
                        fiber_surface=fiber_surface,
                        seeds_list=seeds_list,
                        pred_list=pred_list,
                    )

                    if (not fem_is_valid) and cfg.skip_bad_fem_steps:
                        self._print_fem_failure(step)

                    if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
                        fem_status = "OK" if fem_is_valid else f"BAD({fem_failure_reason})"
                        tqdm.write(
                            f"[{step:05d}] | "
                            f"active(total/mean)={active_count_total:.0f}/{active_count_mean:.2f} | "
                            f"hard_gate={'ON' if use_hard_gate_mask_step else 'off'} | "
                            f"gate_raw(min/mean/max)={gate_raw_min_global:.3f}/{gate_raw_mean_global:.3f}/{gate_raw_max_global:.3f} | "
                            f"gate_sharp(min/mean/max)={gate_sharp_min_global:.3f}/{gate_sharp_mean_global:.3f}/{gate_sharp_max_global:.3f} | "
                            f"gate_dec(min/mean/max)={gate_decoder_min_global:.3f}/{gate_decoder_mean_global:.3f}/{gate_decoder_max_global:.3f} | "
                            f"L_total={row['L_total']:.4e} | "
                            f"L_vol={row['loss_vol']:.3e} "
                            f"L_fem={row['loss_fem']:.3e} "
                            f"L_strut={row['loss_strut']:.3e} "
                            f"L_rep={row['loss_rep']:.3e} "
                            f"L_bnd={row['loss_bnd']:.3e} "
                            f"L_gate={row['loss_gate']:.3e} "
                            f"L_gbin={row['loss_gate_binary']:.3e} | "
                            f"vol={row['vol_frac']:.3f} "
                            f"vol_eff={row['vol_frac_eff']:.3f} "
                            f"(/{cfg.target_volfrac:.3f}) "
                            f"comp={row['comp']:.3e} | "
                            f"w={row['w_geo_mean']:.3e} "
                            f"h={row['h_mean']:.3e} | "
                            f"bw={row['boundary_width_mean']:.3e} "
                            f"ba={row['boundary_alpha_mean']:.3e} "
                            f"bb={row['boundary_beta_mean']:.3e} | "
                            f"theta={row['theta_mean']:.3e} "
                            f"a={row['a_metric_mean']:.3e} | "
                            f"Lse={row['loss_strut_edge']:.3e} "
                            f"Lsv={row['loss_strut_void']:.3e} | "
                            f"rho(min/mean/max)={rho_min:.3f}/{rho_mean:.3f}/{rho_max:.3f} "
                            f"rho_b(min/mean/max)={rho_boundary_min:.3f}/{rho_boundary_mean:.3f}/{rho_boundary_max:.3f} "
                            f"rho_v(min/mean/max)={rho_v_min:.3f}/{rho_v_mean:.3f}/{rho_v_max:.3f} | "
                            f"Δrho={drho:.2e} Δseed={dseed:.2e} grad_mean={g_mean:.2e} | "
                            f"fem={fem_status} | "
                            f"best={best_score:.4e}@{best_step}"
                        )

                    if step >= cfg.early_stop_start and steps_since_improve >= cfg.patience:
                        tqdm.write(
                            f"Early stopping at step {step} | "
                            f"best_step={best_step} | best_score={best_score:.6f}"
                        )
                        break

        # ------------------------------------------------------------
        # Fallback best state
        # ------------------------------------------------------------
        if best_rho is None:
            with torch.no_grad():
                best_rho = rho.detach().clone()
                best_seeds = [s.detach().clone() for s in seeds_list]
                best_pred = [
                    {
                        "face_id": p["face_id"],
                        "seeds_raw": p["seeds_raw"].detach().clone(),
                        "w_raw": p["w_raw"].detach().clone(),
                        "h_raw": None if p["h_raw"] is None else p["h_raw"].detach().clone(),
                        "gate_logits": None if p.get("gate_logits") is None else p["gate_logits"].detach().clone(),
                        "gate_probs_raw": None if p.get("gate_probs_raw") is None else p["gate_probs_raw"].detach().clone(),
                        "gate_probs_sharp": None if p.get("gate_probs_sharp") is None else p["gate_probs_sharp"].detach().clone(),
                        "gate_probs_decoder": None if p.get("gate_probs_decoder") is None else p["gate_probs_decoder"].detach().clone(),
                        "gate_probs": None if p["gate_probs"] is None else p["gate_probs"].detach().clone(),
                        "theta": None if p["theta"] is None else p["theta"].detach().clone(),
                        "a_raw": None if p["a_raw"] is None else p["a_raw"].detach().clone(),
                        "boundary_width_raw": None if p["boundary_width_raw"] is None else p["boundary_width_raw"].detach().clone(),
                        "boundary_alpha_raw": None if p["boundary_alpha_raw"] is None else p["boundary_alpha_raw"].detach().clone(),
                        "boundary_beta_raw": None if p["boundary_beta_raw"] is None else p["boundary_beta_raw"].detach().clone(),
                        "w_geo": p["w_geo"].detach().clone(),
                        "h": None if p["h"] is None else p["h"].detach().clone(),
                        "boundary_width": None if p["boundary_width"] is None else p["boundary_width"].detach().clone(),
                        "boundary_alpha": None if p["boundary_alpha"] is None else p["boundary_alpha"].detach().clone(),
                        "boundary_beta": None if p["boundary_beta"] is None else p["boundary_beta"].detach().clone(),
                        "theta_mean": None if p["theta_mean"] is None else p["theta_mean"].detach().clone(),
                        "a_metric": None if p["a_metric"] is None else p["a_metric"].detach().clone(),
                    }
                    for p in pred_list
                ]
                best_step = step
                best_score = float("inf") if not self._scalar_tensor_is_finite(L_total) else float(L_total.detach().item())

                if best_vol_frac is None:
                    best_vol_frac = float(vol_frac_eff.detach().item())
                if best_comp is None:
                    best_comp = float(comp_val.detach().item())
                if best_w_geo is None:
                    best_w_geo = float(w_geo_mean.detach().item())

        # ------------------------------------------------------------
        # Final outputs
        # ------------------------------------------------------------
        with torch.no_grad():
            final_shape_density = best_rho.clone()
            final_shape_fiber_direction = (
                None if best_fiber_surface is None else best_fiber_surface.clone()
            )
            seed_points_final = self._seed_points_xyz_all_faces(best_seeds, face_tensors)

            if mid_shape_density is None:
                mid_shape_density = final_shape_density.clone()
                seed_points_mid = seed_points_final

        tqdm.write(
            f"FINAL RETURNED: best_step={best_step}, best_score={best_score:.6f} | "
            f"vol_eff={best_vol_frac:.3e}, comp={best_comp:.3e}, w_geo={best_w_geo:.3e}"
        )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        if cfg.MakeTimelaps:
            try:
                recorder.build_video()
            except Exception as e:
                tqdm.write(f"Failed to build timelapse video: {e}")

        return {
            "decoders": decoders,
            "ppnets": ppnets,
            "optimizer": opt,
            "history": history,
            "best_score": best_score,
            "best_step": best_step,
            "best_rho": best_rho,
            "best_seeds": best_seeds,
            "best_pred": best_pred,
            "Initial_shape_density": initial_shape_density,
            "Mid_shape_density": mid_shape_density,
            "Final_shape_density": final_shape_density,
            "Final_shape_fiber_direction": final_shape_fiber_direction,
            "seed_points_init": seed_points_init,
            "seed_points_mid": seed_points_mid,
            "seed_points_final": seed_points_final,
            "A_v": A_v,
            "uv_init_list": uv_init_list,
            "face_tensors": face_tensors,
            "fem_debug_history": self.fem_debug_history,
            "last_fem_debug": self.last_fem_debug,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "shape_path": shape_path,
        }
