from dataclasses import dataclass
import math
import os
import shutil
import time
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

try:
    from .Loss_Boundary import Loss_Boundary
    from .Loss_FEM import Loss_FEM
    from .Loss_SeedActive import Loss_SeedActive
    from .Loss_Volume import Loss_Volume
    from .Loss_Wactive import Loss_Wactive
    from .Loss_rep import Loss_rep
    from .Loss_strut import Loss_strut
except ImportError:
    from Loss_Boundary import Loss_Boundary
    from Loss_FEM import Loss_FEM
    from Loss_SeedActive import Loss_SeedActive
    from Loss_Volume import Loss_Volume
    from Loss_Wactive import Loss_Wactive
    from Loss_rep import Loss_rep
    from Loss_strut import Loss_strut


@dataclass
class TrainingConfig:
    seed_number: int = 15
    LoadingCasee: str = "Unspecified loading case"
    use_Metric_anisotropy: bool = True
    fixed_height: float | None = None
    target_volfrac: float = 0.5
    seed_repulsion_sigma: float = 0.08
    boundary_margin: float = 0.05
    freeze_w: bool = False
    use_boundary_attachment: bool = True
    boundary_volume_assist: float = 0.10
    w_const: float = 0.25

    # boundary defaults / fallback values used by decoder if PPNet does not predict them
    boundary_attach_width: float = 0.03
    boundary_attach_beta: float = 0.01
    boundary_attach_alpha: float = 0.35

    hollow_void_threshold: float = 0.85
    hollow_edge_threshold: float = 0.45
    hollow_temp: float = 0.05
    hollow_rho_edge_min: float = 0.75

    boundary_attach_width_min: float = 0.005
    boundary_attach_width_max: float = 0.10
    duplicate_merge_sigma:float = 0.05

    boundary_attach_alpha_min: float = 0.05
    boundary_attach_alpha_max: float = 1.00

    boundary_attach_beta_min: float = 0.003
    boundary_attach_beta_max: float = 0.05

    predict_tau: bool = None

    w_min: float = 0.005
    w_max_ratio: float = 0.5  # Deprecated: pairwise widths now use 0.8 * seed distance as the upper bound.

    lam_fem: float = 1.0
    lam_vol: float = 2.0
    lam_rep: float = 0.5
    lam_bnd: float = 0.5

    lam_strut: float = 0.02
    lam_strut_edge: float = 1.0
    lam_strut_void: float = 0.25
    lam_width_active: float = 0.05
    lam_seed_active: float = 0.0
    width_target_frac: float = 0.20
    width_target_sparse_boost: float = 1.5
    width_target_frac_max: float = 0.85
    width_warmup_start_frac: float = 0.50
    width_warmup_ramp_frac: float = 0.20
    lam_width_active_hard_multiplier: float = 8.0
    decoder_raw_temp: float = 1.25
    use_band_weighted_fiber_pairs: bool = True
    fiber_band_prior_power: float = 2.0
    fiber_band_prior_floor: float = 0.05
    w_head_bias_init: float | None = None

    comp_normalize_by: float | None = 1e10
    normalize_losses: bool = True
    fem_density_floor: float = 0.02
    skip_bad_fem_steps: bool = True

    num_steps: int = 10000
    context_vector_size: int = 8

    tau: float = 0.02
    tau_pred_start: float = 0.02
    tau_pred_min: float = 1e-4
    tau_pred_max: float = 0.2
    tau_anneal_final: float | None = None
    tau__anneal_final: float | None = None
    tau_anneal_start_frac: float = 0.0
    tau_anneal_ramp_frac: float = 0.5
    beta: float = 0.05
    density_projection_strength: float = 0.0
    density_projection_threshold: float = 0.5
    density_projection_gamma: float = 0.05
    seed_anchor_momentum: float = 0.20
    seed_anchor_warmup_frac: float = 0.05
    use_rolling_seed_anchors: bool = True
    guard_seed_anchor_updates: bool = True
    anchor_guard_rep_max: float = 0.30
    anchor_guard_bnd_max: float = 0.80
    anchor_guard_vol_eff_min: float = 0.10
    anchor_guard_width_factor_min: float = 1.20
    anchor_guard_min_seed_dist_factor: float = 2.0

    collapse_min_seed_dist_factor: float = 2.0
    project_seed_spacing_each_step: bool = True
    seed_projection_iters: int = 4
    allow_seed_outside_domain: bool = True
    allow_seed_outside_domain_warmup_frac: float = 0.50
    seed_domain_margin: float = 0.25
    use_seed_domain_mask: bool = True
    seed_domain_mask_threshold: float = 0.5
    seed_domain_temp: float = 0.05
    seed_domain_mask_support_scale: float = 2.5
    seed_domain_mask_max_points: int = 2048

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

    min_active_seeds: int | None = None
    hard_refine_start_frac: float = 0.85
    freeze_tau_head_during_hard_refine: bool = True
    hard_refine_width_multiplier: float = 2.0

    predict_boundary_params: bool = True

    eps: float = 1e-12

    use_boundary_weighted_volume: bool = False
    boundary_vol_weight: float = 0.20
    effective_volume_power: float = 2.0
    lam_vol_effective: float = 0.5
    lam_vol_sharp: float = 0.5
    sharp_vol_start_frac: float = 0.6
    sharp_vol_ramp_frac: float = 0.3

    Offset_scale: float = 1.00
    seed_offset_scale_start: float | None = None
    seed_offset_scale_final: float | None = None
    seed_offset_scale_ramp_frac: float = 0.60
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
    timelapse_output_folder: str | None = None

    timelapse_frame_step: int = 20
    TM_laps_res_u: int = 100
    TM_laps_res_v: int = 100
    TM_laps_Thr: float = 0.45

    def __post_init__(self):
        if self.tau__anneal_final is not None:
            self.tau_anneal_final = self.tau__anneal_final

        # Backward compatibility: allow legacy absolute-step warmup settings,
        # but prefer the new fraction-based controls.
        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.tau_pred_start <= 0.0:
            raise ValueError(f"tau_pred_start must be > 0, got {self.tau_pred_start}")
        if self.tau_pred_min <= 0.0:
            raise ValueError(f"tau_pred_min must be > 0, got {self.tau_pred_min}")
        if self.tau_pred_max <= self.tau_pred_min:
            raise ValueError(
                f"tau_pred_max must be > tau_pred_min, got min={self.tau_pred_min}, max={self.tau_pred_max}"
            )
        if not (self.tau_pred_min <= self.tau_pred_start <= self.tau_pred_max):
            raise ValueError(
                "tau_pred_start must lie within [tau_pred_min, tau_pred_max], "
                f"got start={self.tau_pred_start}, min={self.tau_pred_min}, max={self.tau_pred_max}"
            )
        if not (0.0 <= self.hard_refine_start_frac <= 1.0):
            raise ValueError(
                f"hard_refine_start_frac must be in [0,1], got {self.hard_refine_start_frac}"
            )
        if self.hard_refine_width_multiplier < 0.0:
            raise ValueError(
                f"hard_refine_width_multiplier must be >= 0, got {self.hard_refine_width_multiplier}"
            )
        if self.tau_anneal_final is not None and self.tau_anneal_final <= 0.0:
            raise ValueError(f"tau_anneal_final must be > 0, got {self.tau_anneal_final}")
        if not (0.0 <= self.density_projection_strength <= 1.0):
            raise ValueError(
                "density_projection_strength must be in [0,1], "
                f"got {self.density_projection_strength}"
            )
        if self.density_projection_gamma <= 0.0:
            raise ValueError(
                f"density_projection_gamma must be > 0, got {self.density_projection_gamma}"
            )
        if not (0.0 <= self.seed_anchor_momentum <= 1.0):
            raise ValueError(
                f"seed_anchor_momentum must be in [0,1], got {self.seed_anchor_momentum}"
            )
        if not (0.0 <= self.seed_anchor_warmup_frac <= 1.0):
            raise ValueError(
                f"seed_anchor_warmup_frac must be in [0,1], got {self.seed_anchor_warmup_frac}"
            )
        if self.anchor_guard_width_factor_min < 1.0:
            raise ValueError(
                "anchor_guard_width_factor_min must be >= 1, "
                f"got {self.anchor_guard_width_factor_min}"
            )
        if self.anchor_guard_min_seed_dist_factor < 0.0:
            raise ValueError(
                "anchor_guard_min_seed_dist_factor must be >= 0, "
                f"got {self.anchor_guard_min_seed_dist_factor}"
            )
        if self.collapse_min_seed_dist_factor < 0.0:
            raise ValueError(
                "collapse_min_seed_dist_factor must be >= 0, "
                f"got {self.collapse_min_seed_dist_factor}"
            )
        if self.seed_projection_iters < 0:
            raise ValueError(f"seed_projection_iters must be >= 0, got {self.seed_projection_iters}")
        if not (0.0 <= self.allow_seed_outside_domain_warmup_frac <= 1.0):
            raise ValueError(
                "allow_seed_outside_domain_warmup_frac must be in [0,1], "
                f"got {self.allow_seed_outside_domain_warmup_frac}"
            )
        if self.seed_domain_margin < 0.0:
            raise ValueError(f"seed_domain_margin must be >= 0, got {self.seed_domain_margin}")
        if not (0.0 <= self.seed_domain_mask_threshold <= 1.0):
            raise ValueError(
                "seed_domain_mask_threshold must be in [0,1], "
                f"got {self.seed_domain_mask_threshold}"
            )
        if self.seed_domain_temp <= 0.0:
            raise ValueError(f"seed_domain_temp must be > 0, got {self.seed_domain_temp}")
        if self.seed_domain_mask_support_scale <= 0.0:
            raise ValueError(
                "seed_domain_mask_support_scale must be > 0, "
                f"got {self.seed_domain_mask_support_scale}"
            )
        if self.seed_domain_mask_max_points < 1:
            raise ValueError(
                "seed_domain_mask_max_points must be >= 1, "
                f"got {self.seed_domain_mask_max_points}"
            )
        if self.seed_offset_scale_start is not None and self.seed_offset_scale_start <= 0.0:
            raise ValueError(
                f"seed_offset_scale_start must be > 0, got {self.seed_offset_scale_start}"
            )
        if self.seed_offset_scale_final is not None and self.seed_offset_scale_final <= 0.0:
            raise ValueError(
                f"seed_offset_scale_final must be > 0, got {self.seed_offset_scale_final}"
            )
        if not (0.0 < self.seed_offset_scale_ramp_frac <= 1.0):
            raise ValueError(
                "seed_offset_scale_ramp_frac must be in (0,1], "
                f"got {self.seed_offset_scale_ramp_frac}"
            )
        if self.min_active_seeds is not None and self.min_active_seeds < 1:
            raise ValueError(f"min_active_seeds must be >= 1, got {self.min_active_seeds}")
        if self.lam_seed_active < 0.0:
            raise ValueError(f"lam_seed_active must be >= 0, got {self.lam_seed_active}")

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
        loading_img=None,
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
        self.loss_volume = Loss_Volume()
        self.loss_fem = Loss_FEM(self)
        self.loss_strut = Loss_strut()
        self.loss_wactive = Loss_Wactive()
        self.loss_boundary = Loss_Boundary()
        self.loss_rep = Loss_rep()
        self.loss_seed_active = Loss_SeedActive()
        self.timelapse_loading_img = (
            None if loading_img is None else self._composite_to_white(np.asarray(loading_img))
        )

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
        self._tb_add_scalar("Loss/SeedActive", row["loss_seed_active"], step)

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
        self._tb_add_scalar("Train/Tau", row["tau"], step)

        self._tb_add_scalar("seed_activation/active_count_total", row["active_count_total"], step)
        self._tb_add_scalar("seed_activation/active_count_mean", row["active_count_mean"], step)
        self._tb_add_scalar("seed_activation/active_frac_mean", row["active_frac_mean"], step)
        self._tb_add_scalar("seed_activation/inactive_count_total", row["inactive_count_total"], step)
        self._tb_add_scalar("seed_activation/inactive_count_mean", row["inactive_count_mean"], step)
        self._tb_add_scalar("seed_activation/inactive_frac_mean", row["inactive_frac_mean"], step)
        self._tb_add_scalar("seed_activation/weight_min", row["seed_active_weight_min"], step)
        self._tb_add_scalar("seed_activation/weight_mean", row["seed_active_weight_mean"], step)
        self._tb_add_scalar("seed_activation/weight_max", row["seed_active_weight_max"], step)
        self._tb_add_scalar("seed_activation/hard_refine_on", row["hard_refine_on"], step)
        self._tb_add_scalar("seed_activation/collapse_active", row["collapse_active"], step)
        self._tb_add_scalar("Loss/WidthActive", row["loss_width_active"], step)
        self._tb_add_scalar("Geometry/lam_width_active_eff", row["lam_width_active_eff"], step)
        self._tb_add_scalar("Train/BestHardScore", row["best_hard_score"], step)
        self._tb_add_scalar("Train/BestHardStep", row["best_hard_step"], step)

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

            seed_active_weight_vals = []
            tau_vals = []
            for p in pred_list:
                if p.get("seed_active_weights") is not None:
                    seed_active_weight_vals.append(p["seed_active_weights"].reshape(-1))
                if p.get("tau") is not None:
                    tau_vals.append(p["tau"].reshape(-1))

            if seed_active_weight_vals:
                self._tb_add_histogram("seed_activation/weights", torch.cat(seed_active_weight_vals, dim=0), step)
            if tau_vals:
                self._tb_add_histogram("Train/TauHist", torch.cat(tau_vals, dim=0), step)

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
        return Loss_Volume.constant_height(
            rho=rho,
            A_v=A_v,
            target_volfrac=target_volfrac,
            eps=eps,
        )

    @staticmethod
    def powered_volume_fraction(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return Loss_Volume.powered_fraction(rho=rho, A_v=A_v, power=power, eps=eps)

    @classmethod
    def volume_loss_powered(
        cls,
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return Loss_Volume().powered(
            rho=rho,
            A_v=A_v,
            target_volfrac=target_volfrac,
            power=power,
            eps=eps,
        )

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

    def seed_offset_scale_for_step(self, step: int) -> float:
        cfg = self.cfg
        start = cfg.Offset_scale if cfg.seed_offset_scale_start is None else cfg.seed_offset_scale_start
        final = start if cfg.seed_offset_scale_final is None else cfg.seed_offset_scale_final
        if cfg.num_steps <= 0:
            return float(final)

        t = min(max(float(step) / max(float(cfg.seed_offset_scale_ramp_frac) * float(cfg.num_steps), 1.0), 0.0), 1.0)
        # Smooth decay: exploration changes gently instead of snapping at a milestone.
        t = t * t * (3.0 - 2.0 * t)
        return float((1.0 - t) * float(start) + t * float(final))

    def allow_seed_outside_domain_for_step(self, step: int) -> bool:
        cfg = self.cfg
        if not bool(cfg.allow_seed_outside_domain):
            return False
        warmup_step = int(round(float(cfg.allow_seed_outside_domain_warmup_frac) * float(cfg.num_steps)))
        return int(step) >= warmup_step

    @staticmethod
    def min_pairwise_seed_distance(seeds_list: list[torch.Tensor]) -> float:
        min_seed_dist = float("inf")
        for seeds_i in seeds_list:
            if seeds_i.shape[0] < 2:
                continue
            d_seed = torch.cdist(seeds_i, seeds_i)
            eye = torch.eye(
                seeds_i.shape[0],
                dtype=torch.bool,
                device=seeds_i.device,
            )
            d_seed = d_seed.masked_fill(eye, float("inf"))
            min_seed_dist = min(min_seed_dist, float(d_seed.min().detach().item()))
        if not math.isfinite(min_seed_dist):
            min_seed_dist = 0.0
        return min_seed_dist

    @staticmethod
    def project_seed_spacing(
        seeds_list: list[torch.Tensor],
        min_dist: float,
        iters: int = 8,
        eps_uv: float = 1e-4,
        detach: bool = True,
        clamp_to_domain: bool = True,
    ) -> list[torch.Tensor]:
        repaired = [(s.detach() if detach else s).clone() for s in seeds_list]
        if min_dist <= 0.0 or iters <= 0:
            return repaired

        for _ in range(int(iters)):
            for seeds in repaired:
                s = int(seeds.shape[0])
                if s < 2:
                    continue
                for i in range(s - 1):
                    for j in range(i + 1, s):
                        diff = seeds[i] - seeds[j]
                        dist = torch.linalg.norm(diff)
                        shortfall = float(min_dist) - float(dist.detach().item())
                        if shortfall <= 0.0:
                            continue
                        if float(dist.detach().item()) > 1e-8:
                            direction = diff / dist.clamp_min(1e-8)
                        else:
                            # Deterministic fallback direction for exact overlaps.
                            angle = torch.as_tensor(
                                2.399963229728653 * float(i + 1) + 1.61803398875 * float(j + 1),
                                dtype=seeds.dtype,
                                device=seeds.device,
                            )
                            direction = torch.stack((torch.cos(angle), torch.sin(angle)))
                        step = 0.5 * shortfall * direction
                        seeds[i] = seeds[i] + step
                        seeds[j] = seeds[j] - step
                if clamp_to_domain:
                    seeds.clamp_(float(eps_uv), 1.0 - float(eps_uv))
        return repaired

    @staticmethod
    def active_width_loss(
        w_raw: torch.Tensor,
        seeds: torch.Tensor,
        seed_active_weights: torch.Tensor | None,
        width_target_frac: float,
        width_target_sparse_boost: float,
        width_target_frac_max: float,
        active_threshold: float,
        raw_temp: float,
        w_min: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        return Loss_Wactive()(
            w_raw=w_raw,
            seeds=seeds,
            seed_active_weights=seed_active_weights,
            width_target_frac=width_target_frac,
            width_target_sparse_boost=width_target_sparse_boost,
            width_target_frac_max=width_target_frac_max,
            active_threshold=active_threshold,
            raw_temp=raw_temp,
            w_min=w_min,
            eps=eps,
        )

    def _tau_for_step(self, step: int) -> float:
        cfg = self.cfg
        if cfg.predict_tau:
            return float(cfg.tau_pred_start)
        if cfg.predict_tau == None:
            return float(cfg.tau)
        

        tau_start = float(cfg.tau)
        tau_end = tau_start if cfg.tau_anneal_final is None else float(cfg.tau_anneal_final)

        anneal = self.ramp_weight(
            step=step,
            total_steps=cfg.num_steps,
            start_frac=cfg.tau_anneal_start_frac,
            ramp_frac=cfg.tau_anneal_ramp_frac,
        )
        return (1.0 - anneal) * tau_start + anneal * tau_end

    def _fallback_tau_value(self) -> float:
        cfg = self.cfg
        if cfg.predict_tau:
            return float(cfg.tau_pred_start)
        return float(cfg.tau)

    @staticmethod
    def _format_elapsed_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        total = int(round(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d} h {minutes:02d} min {secs:02d} sec"
        return f"{minutes:d} min {secs:02d} sec"

    def _timelapse_geometry_summary(self, face_tensors) -> str:
        if self.shell_problem is not None and getattr(self.shell_problem, "mesh", None) is not None:
            fem_mesh = self.shell_problem.mesh
            fem_elems = int(fem_mesh["nelx"]) * int(fem_mesh["nely"]) * int(fem_mesh["nelz"])
        else:
            fem_elems = 0

        surface_pts = int(sum(int(ft["points_xyz"].shape[0]) for ft in face_tensors))

        if self.shell_problem is not None and getattr(self.shell_problem, "brep_bbox", None) is not None:
            bbox = self.shell_problem.brep_bbox
            bbox_dims = (
                float(bbox["xmax"] - bbox["xmin"]),
                float(bbox["ymax"] - bbox["ymin"]),
                float(bbox["zmax"] - bbox["zmin"]),
            )
        else:
            xyz_all = torch.cat([ft["points_xyz"].detach() for ft in face_tensors], dim=0)
            bbox_t = xyz_all.amax(dim=0) - xyz_all.amin(dim=0)
            bbox_dims = tuple(float(v) for v in bbox_t.detach().cpu().tolist())

        load_value = (
            float(getattr(self.shell_problem, "Load_magnitude", 0.0))
            if self.shell_problem is not None
            else 0.0
        )

        bbox_text = " x ".join(f"{dim:.4g}" for dim in bbox_dims)
        return (
            f"BBox: {bbox_text}, "
            f"F={load_value:.6g}, "
            f"SurfacePts={surface_pts}, "
            f"FEM elements={fem_elems}"
        )

    def _timelapse_optimized_parameter_summary(self) -> str:
        cfg = self.cfg
        params = [
            f"seed positions ({int(cfg.seed_number)})",
            "pairwise width" if not cfg.freeze_w else f"width fixed={float(cfg.w_const):.6g}",
        ]

        if cfg.fixed_height is None:
            params.append("height")
        else:
            params.append(f"height fixed={float(cfg.fixed_height):.6g}")

        if cfg.predict_tau:
            params.append("tau")
        else:
            params.append(f"tau fixed/annealed={self._fallback_tau_value():.6g}")

        if cfg.predict_boundary_params:
            params.append("boundary width/alpha/beta")
        elif cfg.use_boundary_attachment:
            params.append("boundary attachment fixed")

        if cfg.use_Metric_anisotropy:
            params.append("metric anisotropy theta/a")

        return "Optimized: " + ", ".join(params)

    @staticmethod
    def _clone_pred_list(pred_list: list[dict]) -> list[dict]:
        return [
            {
                "face_id": p["face_id"],
                "seeds_raw": p["seeds_raw"].detach().clone(),
                "w_raw": p["w_raw"].detach().clone(),
                "h_raw": None if p["h_raw"] is None else p["h_raw"].detach().clone(),
                "seed_active_weights": p["seed_active_weights"].detach().clone(),
                "seed_active_mask": p["seed_active_mask"].detach().clone(),
                "inactive_seed_indices": p["inactive_seed_indices"].detach().clone(),
                "theta": None if p["theta"] is None else p["theta"].detach().clone(),
                "a_raw": None if p["a_raw"] is None else p["a_raw"].detach().clone(),
                "tau": None if p.get("tau") is None else p["tau"].detach().clone(),
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
    @staticmethod
    def volume_loss_with_boundary_discount(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        rho_boundary: torch.Tensor,
        target_volfrac: float,
        boundary_weight: float = 0.20,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return Loss_Volume.with_boundary_discount(
            rho=rho,
            A_v=A_v,
            rho_boundary=rho_boundary,
            target_volfrac=target_volfrac,
            boundary_weight=boundary_weight,
            eps=eps,
        )

    @staticmethod
    def seed_repulsion_term(
        seeds: torch.Tensor,
        seed_active_weights: torch.Tensor | None = None,
        sigma: float = 0.08,
        min_dist: float | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return Loss_rep()(
            seeds=seeds,
            seed_active_weights=seed_active_weights,
            sigma=sigma,
            min_dist=min_dist,
            eps=eps,
        )

    @staticmethod
    def boundary_repulsion_term(
        seeds: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        seed_active_weights: torch.Tensor | None = None,
        margin: float = 0.05,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return Loss_Boundary()(
            seeds=seeds,
            boundary_uv=boundary_uv,
            seed_active_weights=seed_active_weights,
            margin=margin,
            eps=eps,
        )

    @staticmethod
    def compliance_loss(
        comp: torch.Tensor,
        normalize_by: float | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return Loss_FEM.compliance(comp=comp, normalize_by=normalize_by, eps=eps)

    @staticmethod
    def hollow_cell_loss(
        rho: torch.Tensor,
        w_soft: torch.Tensor,
        rho_b: torch.Tensor | None = None,
        void_threshold: float = 0.85,
        edge_threshold: float = 0.45,
        temp: float = 0.05,
        rho_edge_min: float = 0.75,
        lam_void: float = 2.0,
        lam_edge: float = 1.0,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return Loss_strut.with_components(
            rho=rho,
            w_soft=w_soft,
            rho_b=rho_b,
            void_threshold=void_threshold,
            edge_threshold=edge_threshold,
            temp=temp,
            rho_edge_min=rho_edge_min,
            lam_void=lam_void,
            lam_edge=lam_edge,
            eps=eps,
        )

    @staticmethod
    def _scalar_tensor_is_finite(x: torch.Tensor | float | int) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).reshape(()).detach().item())
        return math.isfinite(float(x))

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
        return self.loss_fem.evaluate(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            comp_normalize_by=comp_normalize_by,
            density_floor=density_floor,
            eps=eps,
            save_debug_history=save_debug_history,
        )

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
        seed_active_weights: torch.Tensor | None = None,
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
        zero = torch.zeros((), dtype=rho.dtype, device=rho.device)

        if w_vol != 0.0:
            loss_vol = self.loss_volume(
                rho=rho,
                A_v=A_v,
                target_volfrac=target_volfrac,
                eps=eps,
            )
        else:
            loss_vol = zero

        if w_seed != 0.0:
            loss_seed = self.loss_rep(
                seeds=seeds,
                seed_active_weights=seed_active_weights,
                sigma=sigma,
                eps=eps,
            )
        else:
            loss_seed = zero

        if w_boundary != 0.0:
            loss_boundary = self.loss_boundary(
                seeds=seeds,
                boundary_uv=boundary_uv,
                seed_active_weights=seed_active_weights,
                margin=margin,
                eps=eps,
            )
        else:
            loss_boundary = zero

        loss_strut = zero
        loss_strut_edge = zero
        loss_strut_void = zero

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

            fem_out = self.loss_fem.evaluate(
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
    ):
        face_u_periodic = torch.tensor([bool(u_periodic)], dtype=torch.bool, device=device)
        face_v_periodic = torch.tensor([bool(v_periodic)], dtype=torch.bool, device=device)
        seed_face_id = torch.zeros(seed_number, dtype=torch.long, device=device)

        decoder = self.decoder_cls(
            n_seeds=seed_number,
            use_Metric_anisotropy=self.cfg.use_Metric_anisotropy,
            boundary_solid_idx=boundary_idx_ring1,
            seed_face_id=seed_face_id,
            face_u_periodic=face_u_periodic,
            face_v_periodic=face_v_periodic,
            w_min=self.cfg.w_min,
            w_max_ratio=self.cfg.w_max_ratio,
            beta=self.cfg.beta,
            density_projection_strength=self.cfg.density_projection_strength,
            density_projection_threshold=self.cfg.density_projection_threshold,
            density_projection_gamma=self.cfg.density_projection_gamma,
            duplicate_merge_sigma=self.cfg.duplicate_merge_sigma,
            raw_temp=self.cfg.decoder_raw_temp,
            use_band_weighted_fiber_pairs=self.cfg.use_band_weighted_fiber_pairs,
            fiber_band_prior_power=self.cfg.fiber_band_prior_power,
            fiber_band_prior_floor=self.cfg.fiber_band_prior_floor,
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
            context_dim=self.cfg.context_vector_size,
            n_seeds=seed_number,
            use_Metric_anisotropy=self.cfg.use_Metric_anisotropy,
            predict_boundary_params=self.cfg.predict_boundary_params,
            predict_tau=self.cfg.predict_tau,
            tau_pred_start=self.cfg.tau_pred_start,
            tau_pred_min=self.cfg.tau_pred_min,
            tau_pred_max=self.cfg.tau_pred_max,
            predict_height=(self.cfg.fixed_height is None),
            freeze_w=self.cfg.freeze_w,
            w_const=self.cfg.w_const,   
            w_head_bias_init=(
                float(self.cfg.decoder_raw_temp)
                * math.log(
                    max(min(float(self.cfg.width_target_frac), 1.0 - 1e-4), 1e-4)
                    / max(1.0 - min(float(self.cfg.width_target_frac), 1.0 - 1e-4), 1e-4)
                )
                if self.cfg.w_head_bias_init is None
                else float(self.cfg.w_head_bias_init)
            ),
            allow_seed_outside_domain=(
                bool(self.cfg.allow_seed_outside_domain)
                and float(self.cfg.allow_seed_outside_domain_warmup_frac) <= 0.0
            ),
            seed_domain_margin=self.cfg.seed_domain_margin,
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
            )
            decoders.append(decoder)
            ppnets.append(ppnet)

        return decoders, ppnets

    def _build_optimizer(self, ppnets, decoders):
        cfg = self.cfg
        param_groups = []

        for ppnet in ppnets:
            seed_refine_params = list(ppnet.seed_refine.parameters())
            if getattr(ppnet, "seed_id_embed", None) is not None:
                seed_refine_params.extend(ppnet.seed_id_embed.parameters())

            param_groups.extend([
                {"params": seed_refine_params, "lr": cfg.lr_seed_refine},
                {"params": ppnet.delta_head.parameters(), "lr": cfg.lr_delta_head},
                {"params": ppnet.mlp.parameters(), "lr": cfg.lr_mlp},
            ])

            w_head = getattr(ppnet, "w_head", None)
            if w_head is not None:
                param_groups.append({"params": w_head.parameters(), "lr": cfg.lr_w_head})

            h_head = getattr(ppnet, "h_head", None)
            if (self.cfg.fixed_height is None) and h_head is not None:
                param_groups.append({"params": h_head.parameters(), "lr": cfg.lr_h_head})

            if cfg.use_Metric_anisotropy:
                theta_head = getattr(ppnet, "theta_head", None)
                if theta_head is not None:
                    param_groups.append({"params": theta_head.parameters(), "lr": cfg.lr_mlp})
                a_head = getattr(ppnet, "a_head", None)
                if a_head is not None:
                    param_groups.append({"params": a_head.parameters(), "lr": cfg.lr_mlp})

            if cfg.predict_boundary_params:
                boundary_width_head = getattr(ppnet, "boundary_width_head", None)
                if boundary_width_head is not None:
                    param_groups.append({
                        "params": boundary_width_head.parameters(),
                        "lr": cfg.lr_boundary_heads,
                    })
                boundary_alpha_head = getattr(ppnet, "boundary_alpha_head", None)
                if boundary_alpha_head is not None:
                    param_groups.append({
                        "params": boundary_alpha_head.parameters(),
                        "lr": cfg.lr_boundary_heads,
                    })
                boundary_beta_head = getattr(ppnet, "boundary_beta_head", None)
                if boundary_beta_head is not None:
                    param_groups.append({
                        "params": boundary_beta_head.parameters(),
                        "lr": cfg.lr_boundary_heads,
                    })

            tau_head = getattr(ppnet, "tau_head", None)
            if cfg.predict_tau and tau_head is not None:
                param_groups.append({
                    "params": tau_head.parameters(),
                    "lr": cfg.lr_mlp,
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

    def _finite_or_default(self, x: torch.Tensor | float | int, default: float = float("nan")) -> float:
        if self._scalar_tensor_is_finite(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().item())
            return float(x)
        return default

    @staticmethod
    def _named_trainable_params(modules):
        for mi, module in enumerate(modules):
            for pn, p in module.named_parameters():
                if p.requires_grad:
                    yield mi, pn, p

    @classmethod
    def _nonfinite_grad_info(cls, modules):
        bad = []
        for mi, pn, p in cls._named_trainable_params(modules):
            g = p.grad
            if g is not None and not torch.isfinite(g).all():
                bad.append((mi, pn))
        return bad

    @classmethod
    def _nonfinite_param_info(cls, modules):
        bad = []
        for mi, pn, p in cls._named_trainable_params(modules):
            if not torch.isfinite(p).all():
                bad.append((mi, pn))
        return bad

    @staticmethod
    def _restore_param_snapshot(snapshot):
        for p, saved in snapshot.items():
            p.data.copy_(saved)

    @staticmethod
    def _clear_optimizer_state_for_params(opt, params):
        for p in params:
            if p in opt.state:
                opt.state.pop(p, None)

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

    def _seed_domain_mask_for_face(self, ft):
        cfg = self.cfg
        if not bool(cfg.use_seed_domain_mask):
            return None
        cached = ft.get("_seed_domain_mask_callable", None)
        if cached is not None:
            return cached
        mask_grid = ft.get("seed_domain_mask_grid", None)
        if mask_grid is not None:
            return mask_grid

        uv_face = ft.get("seed_domain_uv_support", ft["uv"])
        if uv_face.numel() == 0:
            return None

        uv_support = uv_face.detach()
        max_points = int(cfg.seed_domain_mask_max_points)
        if uv_support.shape[0] > max_points:
            sample_idx = torch.linspace(
                0,
                uv_support.shape[0] - 1,
                max_points,
                device=uv_support.device,
            ).round().to(torch.long)
            uv_support = uv_support[sample_idx]

        sigma_value = ft.get("seed_domain_sigma", None)
        if sigma_value is None:
            sigma = self._estimate_uv_mask_tol(
                uv_support,
                u_periodic=bool(ft.get("u_periodic", False)),
                v_periodic=bool(ft.get("v_periodic", False)),
                fallback=float(cfg.boundary_margin),
                scale=float(cfg.seed_domain_mask_support_scale),
            )
        elif torch.is_tensor(sigma_value):
            sigma = float(sigma_value.detach().cpu().item())
        else:
            sigma = float(sigma_value)
        sigma = max(float(sigma), float(cfg.eps))
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        def mask_fn(seeds):
            support = uv_support.to(device=seeds.device, dtype=seeds.dtype)
            diff = seeds.unsqueeze(1) - support.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)
            dmin = torch.norm(diff, dim=-1).amin(dim=1)
            sigma_t = torch.as_tensor(sigma, device=seeds.device, dtype=seeds.dtype)
            return torch.exp(-0.5 * (dmin / sigma_t.clamp_min(cfg.eps)).pow(2))

        ft["_seed_domain_mask_callable"] = mask_fn
        return mask_fn

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
    ):
        cache = []

        for ft in face_tensors:
            device = ft["uv"].device
            uv_face = ft["uv"]
            uv_dense = ft["uv"]
            xyz_dense = ft["points_xyz"]
            Xu_dense = ft["Xu"]
            Xv_dense = ft["Xv"]

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
                "seed_domain_mask": self._seed_domain_mask_for_face(ft),
                "faces_ijk": ft["faces_ijk"],
            })

        return cache

    def evaluate_cached_face_fields(self, render_cache, decoder, pred):
        tau = self._fallback_tau_value() if pred.get("tau") is None else pred["tau"]
        decoder_out = decoder.evaluate_at_uv(
            points_uv=render_cache["uv_dense"],
            Xu=render_cache["Xu_dense"],
            Xv=render_cache["Xv_dense"],
            tau=tau,
            seeds_raw=pred["seeds_raw"],
            w_raw=pred["w_raw"],
            h_raw=pred.get("h_raw", None),
            theta=pred.get("theta", None),
            a_raw=pred.get("a_raw", None),
            points_face_id=render_cache["local_face_id"],
            boundary_uv=render_cache["boundary_uv"],
            boundary_face_id=render_cache["boundary_face_id"],
            boundary_width_raw=pred.get("boundary_width_raw", None),
            boundary_alpha_raw=pred.get("boundary_alpha_raw", None),
            boundary_beta_raw=pred.get("boundary_beta_raw", None),
            hard_seed_mask=True,
            seed_domain_mask=render_cache.get("seed_domain_mask", None),
            seed_domain_mask_threshold=self.cfg.seed_domain_mask_threshold,
            seed_domain_temp=self.cfg.seed_domain_temp,
        )

        return {
            "xyz_dense": render_cache["xyz_dense"],
            "rho_dense": decoder_out["rho"],
            "fiber3d_dense": decoder_out["fiber3d"],
            "faces_ijk": render_cache["faces_ijk"],
        }

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
        loading_img=None,
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
            faces_local = out["faces_ijk"].detach().cpu().numpy().astype(np.int64)
            if faces_local.size > 0:
                pv_faces_all = np.empty((faces_local.shape[0], 4), dtype=np.int64)
                pv_faces_all[:, 0] = 3
                pv_faces_all[:, 1:] = faces_local
                mesh_all = pv.PolyData(xyz, pv_faces_all.reshape(-1))
                mesh_all["rho"] = rho_dense.astype(np.float32)
                density_meshes.append(mesh_all)

            if faces_local.size > 0:
                solid_keep = np.all(rho_dense[faces_local] >= float(thr), axis=1)
                faces_solid_local = faces_local[solid_keep]
                if faces_solid_local.size > 0:
                    pv_faces_solid = np.empty((faces_solid_local.shape[0], 4), dtype=np.int64)
                    pv_faces_solid[:, 0] = 3
                    pv_faces_solid[:, 1:] = faces_solid_local
                    solid_meshes.append(pv.PolyData(xyz, pv_faces_solid.reshape(-1)))

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

        seed_vis = self._seed_points_xyz_and_activity_all_faces(
            seeds_list=seeds_list,
            pred_list=pred_list,
            face_tensors=self.current_face_tensors,
        )
        active_seed_points = seed_vis["xyz_active"]
        inactive_seed_points = seed_vis["xyz_inactive"]
        seed_point_size = max(6.0, 0.006 * max(diag, 1.0) * 100.0)
        show_seed_points = True
        show_axes_widget = True

        first_face_density_img = None
        if render_cache:
            first_cache = render_cache[0]
            first_face_id = first_cache["face_id"]
            first_out = self.evaluate_cached_face_fields(
                first_cache,
                dec_by_face_id[first_face_id],
                pred_by_face_id[first_face_id],
            )
            first_seed_idx = 0
            for idx, ft in enumerate(self.current_face_tensors):
                if ft["face_id"] == first_face_id:
                    first_seed_idx = idx
                    break
            first_face_density_img = self._render_first_face_density_2d(
                cache_i=first_cache,
                out_i=first_out,
                seeds_i=seeds_list[first_seed_idx],
                pred_i=pred_by_face_id[first_face_id],
                window_size=(1050, 1050),
            )

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
            ("3D Material Distribution | Front View", "density", "xz"),
            ("3D Material Distribution | Side View", "density", "yz"),
            ("3D Material Distribution | Top View", "density", "xy"),
        ]
        perspective_spec = ("3D Material Distribution | Perspective View", "density", "iso")
        top_window_size = (560, 430)
        bottom_window_size = (1050, 1050)

        top_imgs = []
        if loading_img is not None:
            loading_panel_img = cv2.resize(
                loading_img,
                top_window_size,
                interpolation=cv2.INTER_AREA if loading_img.shape[1] > top_window_size[0] else cv2.INTER_CUBIC,
            )
            top_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        loading_panel_img,
                        "Voxel Loading And Boundary Conditions",
                    )
                )
            )
        for title, mode, view in top_specs:
            pl = make_plotter(title, mode, window_size=top_window_size)
            img = self._render_offscreen_plotter(pl, view)
            top_imgs.append(self._add_panel_border(self._add_image_title(img, title)))
            pl.close()

        bottom_imgs = []
        if first_face_density_img is not None:
            bottom_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        first_face_density_img,
                        f"UV Domain Density Distribution"
                        )
                )
            )
        title, mode, view = perspective_spec
        pl = make_plotter(title, mode, window_size=bottom_window_size)
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

    def _render_first_face_density_2d(
        self,
        cache_i,
        out_i,
        seeds_i,
        pred_i,
        window_size=(820, 820),
    ):
        width, height = int(window_size[0]), int(window_size[1])
        uv = cache_i["uv_dense"].detach().cpu().numpy().astype(np.float64)
        rho = out_i["rho_dense"].detach().cpu().numpy().astype(np.float64)
        faces = cache_i["faces_ijk"].detach().cpu().numpy().astype(np.int64)
        seeds = seeds_i.detach().cpu().numpy().astype(np.float64)

        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="white")
        ax = fig.add_axes([0.08, 0.08, 0.78, 0.84])

        if faces.size > 0:
            tpc = ax.tripcolor(
                uv[:, 0],
                uv[:, 1],
                faces,
                rho,
                shading="gouraud",
                cmap="viridis",
                vmin=0.0,
                vmax=max(1.0, float(np.nanquantile(rho, 0.995)) if rho.size else 1.0),
            )
        else:
            tpc = ax.scatter(
                uv[:, 0],
                uv[:, 1],
                c=rho,
                s=10,
                cmap="viridis",
                vmin=0.0,
                vmax=max(1.0, float(np.nanquantile(rho, 0.995)) if rho.size else 1.0),
                linewidths=0,
            )

        active_values = pred_i.get("seed_active_mask", None)
        if active_values is not None:
            active = active_values.detach().cpu().numpy().reshape(-1).astype(bool)
        else:
            active = np.ones((seeds.shape[0],), dtype=bool)

        if seeds.shape[0] > 0:
            if np.any(~active):
                ax.scatter(
                    seeds[~active, 0],
                    seeds[~active, 1],
                    s=72,
                    c="#6b7280",
                    edgecolors="white",
                    linewidths=1.4,
                    alpha=0.55,
                    zorder=5,
                )
            if np.any(active):
                ax.scatter(
                    seeds[active, 0],
                    seeds[active, 1],
                    s=92,
                    c="#ef4444",
                    edgecolors="white",
                    linewidths=1.6,
                    zorder=6,
                )

        ax.set_xlim(float(np.nanmin(uv[:, 0])), float(np.nanmax(uv[:, 0])))
        ax.set_ylim(float(np.nanmin(uv[:, 1])), float(np.nanmax(uv[:, 1])))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("u", fontsize=11)
        ax.set_ylabel("v", fontsize=11)
        ax.tick_params(labelsize=9, colors="#374151")
        ax.grid(color="white", linewidth=0.6, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")

        cax = fig.add_axes([0.89, 0.12, 0.025, 0.76])
        cb = fig.colorbar(tpc, cax=cax)
        cb.set_label("rho", fontsize=10)
        cb.ax.tick_params(labelsize=9, colors="#374151")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3].copy()
        plt.close(fig)
        return img

    def _seed_points_xyz_and_activity_all_faces(self, seeds_list, pred_list, face_tensors):
        xyz_active = []
        xyz_inactive = []
        active_weight = []
        inactive_weight = []

        for seeds, pred, ft in zip(seeds_list, pred_list, face_tensors):
            xyz_i = self.generator.seeds_uv_to_xyz_nearest(
                seeds,
                ft["uv"],
                ft["points_xyz"],
            )

            active_mask_i = pred.get("seed_active_mask", None)
            active_weights_i = pred.get("seed_active_weights", None)

            if active_mask_i is None:
                xyz_active.append(xyz_i)
                continue

            active_mask = active_mask_i.detach().cpu().numpy().astype(bool)
            weights = (
                active_weights_i.detach().cpu().numpy()
                if active_weights_i is not None
                else active_mask.astype(float)
            )
            participating_mask = active_mask
            inactive_mask = ~participating_mask

            xyz_i_active = xyz_i[participating_mask]
            xyz_i_inactive = xyz_i[inactive_mask]

            if len(xyz_i_active) > 0:
                xyz_active.append(xyz_i_active)
                active_weight.append(weights[participating_mask])

            if len(xyz_i_inactive) > 0:
                xyz_inactive.append(xyz_i_inactive)
                inactive_weight.append(weights[inactive_mask])

        import numpy as np

        xyz_active = np.concatenate(xyz_active, axis=0) if len(xyz_active) > 0 else None
        xyz_inactive = np.concatenate(xyz_inactive, axis=0) if len(xyz_inactive) > 0 else None
        active_weight = np.concatenate(active_weight, axis=0) if len(active_weight) > 0 else None
        inactive_weight = np.concatenate(inactive_weight, axis=0) if len(inactive_weight) > 0 else None

        return {
            "xyz_active": xyz_active,
            "xyz_inactive": xyz_inactive,
            "active_weight": active_weight,
            "inactive_weight": inactive_weight,
        }
    
    def visualize_best_seed_activity(self, result, points_xyz=None, faces_ijk=None):
        best_seeds = result["best_seeds"]
        best_pred = result["best_pred"]
        face_tensors = result["face_tensors"]

        seed_vis = self._seed_points_xyz_and_activity_all_faces(
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
                label="Inactive seeds",
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

        boundary_width_raw = pred.get("boundary_width_raw", None)
        boundary_alpha_raw = pred.get("boundary_alpha_raw", None)
        boundary_beta_raw = pred.get("boundary_beta_raw", None)

        # ------------------------------------------------------------
        # 6) Evaluate decoder on CAD-native dense query points
        # ------------------------------------------------------------
        tau = self._fallback_tau_value() if pred.get("tau") is None else pred["tau"]
        decoder_out = decoder.evaluate_at_uv(
            points_uv=uv_dense,
            Xu=Xu_dense,
            Xv=Xv_dense,
            tau=tau,
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
            hard_seed_mask=True,
            seed_domain_mask=self._seed_domain_mask_for_face(ft),
            seed_domain_mask_threshold=self.cfg.seed_domain_mask_threshold,
            seed_domain_temp=self.cfg.seed_domain_temp,
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

    def Visualize_fresult_final_fiber_Direction_3D(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    def visualize_result_final_fiber_direction_3d(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    def Visualize_fresult_final_fiber_Direction_2D(
        self,
        result,
        points_xyz=None,
        faces_ijk=None,
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

    def visualize_result_final_smooth_surface_pyvista(
        self,
        result,
        points_xyz=None,
        faces_ijk=None,
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

        if shape_or_path is None:
            shape_or_path = result.get("shape_path", None)
        if shape_or_path is None:
            raise ValueError(
                "shape_or_path is required for smooth CAD-native visualization. "
                "Pass it explicitly or store it in result['shape_path']."
            )

        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

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
        per_face = []

        for face_data in dense["per_face"]:
            xyz = face_data["xyz_dense"].detach().cpu().numpy().astype(np.float32)
            rho = face_data["rho_dense"].detach().cpu().numpy().astype(np.float32)
            face_id = face_data["face_id"]
            Nu, Nv = face_data["grid_shape"]
            mask = face_data["mask_dense_valid"].detach().cpu().numpy()
            uv = face_data["uv_dense"].detach().cpu().numpy().astype(np.float32)

            full_indices = -np.ones(mask.shape[0], dtype=np.int64)
            full_indices[mask] = np.arange(mask.sum(), dtype=np.int64)

            faces_keep = []

            def idx(i, j):
                return i * Nv + j

            for i in range(Nu - 1):
                for j in range(Nv - 1):
                    ids = [idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)]
                    mapped = [full_indices[k] for k in ids]
                    if any(m < 0 for m in mapped):
                        continue

                    i0, i1, i2, i3 = mapped
                    if rho[i0] >= thr_used and rho[i1] >= thr_used and rho[i2] >= thr_used:
                        faces_keep.append([i0, i1, i2])
                    if rho[i2] >= thr_used and rho[i1] >= thr_used and rho[i3] >= thr_used:
                        faces_keep.append([i2, i1, i3])

            faces_keep = np.asarray(faces_keep, dtype=np.int64)
            if faces_keep.size == 0:
                per_face.append({
                    "face_id": face_id,
                    "uv": uv,
                    "xyz": xyz,
                    "rho": rho,
                    "faces_keep": faces_keep,
                })
                continue

            pv_faces = np.empty((faces_keep.shape[0], 4), dtype=np.int64)
            pv_faces[:, 0] = 3
            pv_faces[:, 1:] = faces_keep
            mesh = pv.PolyData(xyz, pv_faces.reshape(-1))

            if show_density:
                mesh["rho"] = rho
                plotter.add_mesh(mesh, scalars="rho", cmap="viridis", clim=[0, 1])
            else:
                plotter.add_mesh(mesh, color="lightblue")

            per_face.append({
                "face_id": face_id,
                "uv": uv,
                "xyz": xyz,
                "rho": rho,
                "faces_keep": faces_keep,
            })
        plotter.show()
        return {
            "thr_used": float(thr_used),
            "volfrac_cont": float(volfrac_cont),
            "volfrac_thr": float(volfrac_thr),
            "dense": dense,
            "per_face": per_face,
        }
  
    def train(self, shape_path, face_tensors):
        cfg = self.cfg
        train_start_time = time.perf_counter()

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
        # ------------------------------------------------------------
        # Build global vertex areas
        A_v = torch.zeros((vertices_number,), dtype=dtype, device=device)
        local_vertex_areas = []
        local_face_weights = []


        # in this for loop, we compute the lumped vertex areas for each face and accumulate them into a global vertex area tensor A_v.
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
        # Build initial seeds from face tensors for all faces, which will be optimized during training. Output shape is (n_faces, n_seeds_per_face, 2).
        uv_init_list = self._init_face_seeds(face_tensors)
        uv_anchor_list = [uv_i.clone() for uv_i in uv_init_list]

        contexts = [
            torch.zeros(1, cfg.context_vector_size, device=device, dtype=dtype)
            for _ in face_tensors
        ]
        # Build the optimizer for all ppnet parameters. It includes the learning parameters,  optimizer type and learning rate are determined by the configuration (cfg).
        opt = self._build_optimizer(ppnets, decoders)
        # getatt(A,"S",None) is try to reach attribute "S" in object A, if it doesn't exist, it will return None instead of raising an error. 
        # here we are trying to get the "scheduler_milestones" attribute from the configuration (cfg). I
        #these milestones are specific training steps at which the learning rate will be adjusted according to a predefined schedule. 
        raw_milestones = getattr(cfg, "scheduler_milestones", None)
        if raw_milestones is None:
            milestones = []
        else:
            # isinstance(raw_milestones, (int, float)) checks if raw_milestones is a single number (int or float). 
            raw_seq = [raw_milestones] if isinstance(raw_milestones, (int, float)) else list(raw_milestones)
            milestones = []
            for m in raw_seq:
                m = float(m)
                # Support both fractional milestones (0..1] and absolute step indices (>1).
                step_m = int(round(m * cfg.num_steps)) if m <= 1.0 else int(round(m))
                if 0 < step_m < cfg.num_steps:
                    milestones.append(step_m)
            milestones = sorted(set(milestones))

        #print(f"scheduler_milestones: {milestones}")

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
        timelapse_output_folder = None
        if cfg.MakeTimelaps:
            case_name = shape_path.stem
            timelapse_output_folder = getattr(cfg, "timelapse_output_folder", None)
            if timelapse_output_folder:
                timelapse_output_folder = os.path.normpath(str(timelapse_output_folder))
                os.makedirs(timelapse_output_folder, exist_ok=True)
                frame_out_dir = os.path.join(timelapse_output_folder, "timelapse_frames")
                video_path = os.path.join(timelapse_output_folder, case_name + "_timelapse.avi")
            else:
                frame_out_dir = "timelapse_frames"
                video_path = case_name + "_timelapse.avi"
            # defining the timelapse recorder, which will save the training progress as a video. 
            # The output directory for the frames is "timelapse_frames", 
            # the video will be saved with the name "{case_name}_timelapse.avi". 
            # The frames per second (fps) for the video is set to 8.
            geometry_summary = self._timelapse_geometry_summary(face_tensors)
            recorder = TimelapseRecorder(
                out_dir=frame_out_dir,
                video_path=video_path,
                fps=8,
                header_title=(
                    f"{shape_path.name} ({geometry_summary}) | "
                    f"Loading case: {cfg.LoadingCasee} | "
                    f"Target volfrac: {cfg.target_volfrac:.3f}"
                ),
                header_subtitle=self._timelapse_optimized_parameter_summary(),
            )
            # building a cache for rendering the timelapse, which likely includes precomputing certain data or settings that will be used 
            # repeatedly during the rendering of each frame in the timelapse video. 
            render_cache = self.build_timelapse_render_cache(
                face_tensors=face_tensors,
            )
            if self.timelapse_loading_img is None and self.shell_problem is not None:
                try:
                    self.timelapse_loading_img = self.shell_problem.show_voxels_surface_and_bc(
                        return_img=True,
                        off_screen=True,
                        window_size=(520, 280),
                    )
                    self.timelapse_loading_img = self._composite_to_white(self.timelapse_loading_img)
                except Exception as e:
                    tqdm.write(f"Failed to render timelapse loading panel: {e}")

        # ------------------------------------------------------------
        # Loss normalizers
        # ------------------------------------------------------------
        # These RunningNorm instances are used to keep track of the running mean and standard deviation of various loss components during training.
        # if on , it will normalize the loss components to have a more stable training process, especially when the scales of different loss terms vary significantly.
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
        best_active_count = None
        best_inactive_count = None
        best_rho = None
        best_fiber_surface = None
        best_seeds = None
        best_pred = None
        best_hard_score = float("inf")
        best_hard_vol_frac = None
        best_hard_comp = None
        best_hard_w_geo = None
        best_hard_step = -1
        best_hard_active_count = None
        best_hard_inactive_count = None
        best_hard_rho = None
        best_hard_fiber_surface = None
        best_hard_seeds = None
        best_hard_pred = None
        # ------------------------------------------------------------

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
        anchor_update_allowed = True
        history = []

        self.current_face_tensors = face_tensors

        # ------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------
        # The main traiing loop iterates for a number of steps defined in the configuration (cfg.num_steps).
        # To have a progress bar for training, it uses the tqdm library, which provides a visual representation of the training progress in the console.
        # It is equal to for step in training_steps, but with an added progress bar that shows the current step and other relevant information during training.
        # desc="Training" sets the description of the progress bar to "Training", leave=True keeps the progress bar displayed after completion, and dynamic_ncols=True allows the progress bar to adjust its width dynamically based on the terminal size.
        with tqdm(
            range(cfg.num_steps),
            desc="Training",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for step in pbar:
                
                
                # if cfg.allow_seed_outside_domain is true and the warmup period is over, allow seeds to be placed outside the domain
                allow_seed_outside_domain_step = self.allow_seed_outside_domain_for_step(step)
                for ppnet in ppnets:
                    ppnet.allow_seed_outside_domain = allow_seed_outside_domain_step
                # if cfg.predict_tau is none , always use cfg.tau as the tau value.
                # if cfg.predict_tau is true,  it returns  cfg.tau_predic_start
                # if Cfg.predict_tau is false, it returns annealed value.
                tau_step = self._tau_for_step(step)
                # if cfg.use_hard_refine_step is true, it will return true when the current step is greater than or equal to the step defined by hard_refine_start_frac, which is a fraction of the total number of steps.
                use_hard_refine_step = step >= int(round(float(cfg.hard_refine_start_frac) * float(cfg.num_steps)))
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
                seed_active_terms = []
                w_geo_terms = []
                h_terms = []

                boundary_width_terms = []
                boundary_alpha_terms = []
                boundary_beta_terms = []

                theta_mean_terms = []
                a_metric_terms = []

                width_active_terms = []
                face_weights_this_step = []

                participating_count_total = 0.0
                participating_frac_sum = 0.0
                inactive_count_total = 0.0
                inactive_frac_sum = 0.0
                active_weight_min_list = []
                active_weight_mean_sum = 0.0
                active_weight_max_list = []
                active_face_count = 0

                # Activate losses based on their lambda values in the configuration (cfg). If a lambda value is set to 0.0, the corresponding loss will not be computed during training
                compute_rep_loss = cfg.lam_rep != 0.0
                compute_bnd_loss = cfg.lam_bnd != 0.0
                compute_strut_loss = cfg.lam_strut != 0.0
                compute_vol_loss = cfg.lam_vol != 0.0
                compute_width_active_loss = cfg.lam_width_active != 0.0
                compute_seed_active_loss = cfg.lam_seed_active != 0.0

                # Determine whether to update seed anchors based on the configuration and current step, seed anchors are reference points used in the training process.
                # if it is on, it will update the seed anchors after a certain warmup period, and the update is allowed based on the configuration settings.
                update_seed_anchors = (
                    cfg.use_rolling_seed_anchors
                    and step >= int(round(float(cfg.seed_anchor_warmup_frac) * float(cfg.num_steps)))
                    and (anchor_update_allowed or not cfg.guard_seed_anchor_updates)
                )
                seed_offset_scale_step = self.seed_offset_scale_for_step(step)
                uv_anchor_next_list = []

                for ft, decoder, ppnet, uv_anchor_i, context_i, A_local, face_weight_i in zip(
                    face_tensors,
                    decoders,
                    ppnets,
                    uv_anchor_list,
                    contexts,
                    local_vertex_areas,
                    local_face_weights,
                ):
                    pred_i = ppnet(context_i, uv_anchor_i, offset_scale=seed_offset_scale_step)

                    seeds_raw_i = pred_i["seeds_raw"][0]
                    if cfg.project_seed_spacing_each_step:
                        seeds_raw_i = self.project_seed_spacing(
                            seeds_list=[seeds_raw_i],
                            min_dist=float(cfg.collapse_min_seed_dist_factor) * float(cfg.w_min),
                            iters=int(cfg.seed_projection_iters),
                            detach=False,
                            clamp_to_domain=not allow_seed_outside_domain_step,
                        )[0]
                        pred_i["seeds_raw"] = pred_i["seeds_raw"].clone()
                        pred_i["seeds_raw"][0] = seeds_raw_i
                    w_raw_i = pred_i["w_raw"][0]

                    h_raw_i = None
                    if cfg.fixed_height is None and "h_raw" in pred_i:
                        h_raw_i = pred_i["h_raw"][0]

                    theta_pred_i = pred_i.get("theta", None)
                    theta_i = theta_pred_i[0] if (cfg.use_Metric_anisotropy and theta_pred_i is not None) else None
                    a_raw_pred_i = pred_i.get("a_raw", None)
                    a_raw_i = a_raw_pred_i[0] if (cfg.use_Metric_anisotropy and a_raw_pred_i is not None) else None

                    boundary_width_pred_i = pred_i.get("boundary_width_raw", None)
                    boundary_width_raw_i = boundary_width_pred_i[0] if boundary_width_pred_i is not None else None
                    boundary_alpha_pred_i = pred_i.get("boundary_alpha_raw", None)
                    boundary_alpha_raw_i = boundary_alpha_pred_i[0] if boundary_alpha_pred_i is not None else None
                    boundary_beta_pred_i = pred_i.get("boundary_beta_raw", None)
                    boundary_beta_raw_i = boundary_beta_pred_i[0] if boundary_beta_pred_i is not None else None
                    tau_pred_i = pred_i.get("tau", None)
                    tau_step = tau_pred_i[0] if tau_pred_i is not None else tau_step

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
                    seed_domain_mask_i = self._seed_domain_mask_for_face(ft)

                    decoder_out = decoder(
                        points_uv=ft["uv"],
                        Xu=ft["Xu"],
                        Xv=ft["Xv"],
                        tau=tau_step,
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
                        seed_domain_mask=seed_domain_mask_i,
                        seed_domain_mask_threshold=cfg.seed_domain_mask_threshold,
                        seed_domain_temp=cfg.seed_domain_temp,
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
                            "seed_active_weights",
                            "seed_active_mask",
                        ],
                    )

                    seeds_i = decoder_out["seeds"]
                    rho_i = decoder_out["rho"]
                    rho_s_i = decoder_out["rho_s"]
                    w_geo_i = decoder_out["w_geo"]
                    w_soft_i = decoder_out["w_soft"]
                    fiber3d_i = decoder_out["fiber3d"]
                    rho_v_i = decoder_out["rho_v"]
                    rho_b_i = decoder_out["rho_b"]
                    edge_field_i = decoder_out["edge_field"]
                    seed_active_weights_i = decoder_out["seed_active_weights"]
                    seed_active_mask_i = decoder_out["seed_active_mask"]
                    inactive_seed_indices_i = decoder_out["inactive_seed_indices"]
                    active_count_i = float(seed_active_mask_i.detach().to(torch.float32).sum().item())
                    inactive_count_i = float((~seed_active_mask_i.detach()).to(torch.float32).sum().item())
                    total_seed_i = max(int(seed_active_mask_i.numel()), 1)
                    participating_count_total += active_count_i
                    participating_frac_sum += active_count_i / float(total_seed_i)
                    inactive_count_total += inactive_count_i
                    inactive_frac_sum += inactive_count_i / float(total_seed_i)
                    active_weight_min_list.append(float(seed_active_weights_i.detach().min().item()))
                    active_weight_mean_sum += float(seed_active_weights_i.detach().mean().item())
                    active_weight_max_list.append(float(seed_active_weights_i.detach().max().item()))
                    active_face_count += 1

                    if compute_seed_active_loss:
                        target_active = float(cfg.min_active_seeds or total_seed_i)
                        seed_active_terms.append(
                            self.loss_seed_active(
                                seed_active_weights=seed_active_weights_i,
                                target_active=target_active,
                                eps=cfg.eps,
                            )
                        )

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

                    if compute_width_active_loss:
                        width_active_terms.append(
                            self.loss_wactive(
                                w_raw=w_raw_i,
                                seeds=seeds_i,
                                seed_active_weights=seed_active_weights_i,
                                width_target_frac=cfg.width_target_frac,
                                width_target_sparse_boost=cfg.width_target_sparse_boost,
                                width_target_frac_max=cfg.width_target_frac_max,
                                active_threshold=0.5,
                                raw_temp=cfg.decoder_raw_temp,
                                w_min=cfg.w_min,
                                eps=cfg.eps,
                            )
                        )

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
                    if update_seed_anchors:
                        anchor_alpha = float(cfg.seed_anchor_momentum)
                        uv_anchor_next_i = (
                            (1.0 - anchor_alpha) * uv_anchor_i + anchor_alpha * seeds_i.detach()
                        )
                    else:
                        uv_anchor_next_i = uv_anchor_i.detach().clone()
                    uv_anchor_next_list.append(uv_anchor_next_i)

                    pred_list.append({
                        "face_id": ft["face_id"],
                        "seeds_raw": seeds_raw_i.detach().clone(),
                        "w_raw": w_raw_i.detach().clone(),
                        "h_raw": None if h_raw_i is None else h_raw_i.detach().clone(),
                        "seed_active_weights": seed_active_weights_i.detach().clone(),
                        "seed_active_mask": seed_active_mask_i.detach().clone(),
                        "inactive_seed_indices": inactive_seed_indices_i.detach().clone(),
                        "theta": None if theta_i is None else theta_i.detach().clone(),
                        "a_raw": None if a_raw_i is None else a_raw_i.detach().clone(),
                        "tau": None if tau_pred_i is None else tau_pred_i.detach().clone(),

                        "boundary_width": boundary_width_i.detach().clone() if isinstance(boundary_width_i, torch.Tensor) else boundary_width_i,
                        "boundary_alpha": boundary_alpha_i.detach().clone() if isinstance(boundary_alpha_i, torch.Tensor) else boundary_alpha_i,
                        "boundary_beta": boundary_beta_i.detach().clone() if isinstance(boundary_beta_i, torch.Tensor) else boundary_beta_i,

                        "w_geo": w_geo_i.detach().clone(),
                        "h": h_i.detach().clone() if isinstance(h_i, torch.Tensor) else h_i,

                        "boundary_width_raw": None if boundary_width_raw_i is None else boundary_width_raw_i.detach().clone(),
                        "boundary_alpha_raw": None if boundary_alpha_raw_i is None else boundary_alpha_raw_i.detach().clone(),
                        "boundary_beta_raw": None if boundary_beta_raw_i is None else boundary_beta_raw_i.detach().clone(),

                        "theta_mean": None if theta_i is None else theta_i.mean().detach().clone(),
                        "a_metric": None if a_raw_i is None else (
                            0.5 * (2.0 - 0.5) * torch.tanh(a_raw_i) + 0.5 * (2.0 + 0.5)
                        ).mean().detach().clone(),
                    })

                    if compute_rep_loss:
                        rep_terms.append(
                            self.loss_rep(
                                seeds=seeds_i,
                                seed_active_weights=None,
                                sigma=cfg.seed_repulsion_sigma,
                                min_dist=2.0 * float(cfg.w_min),
                                eps=cfg.eps,
                            )
                        )

                    if compute_bnd_loss:
                        bnd_terms.append(
                            self.loss_boundary(
                                seeds=seeds_i,
                                boundary_uv=boundary_uv_i,
                                seed_active_weights=None,
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
                    if compute_strut_loss:
                        (
                            loss_strut_i,
                            loss_strut_edge_i,
                            loss_strut_void_i,
                            edge_mask_i,
                            void_mask_i,
                        ) = self.loss_strut.with_components(
                            rho=rho_i,
                            w_soft=w_soft_i,
                            rho_b=rho_b_i,
                            void_threshold=cfg.hollow_void_threshold,
                            edge_threshold=cfg.hollow_edge_threshold,
                            temp=cfg.hollow_temp,
                            rho_edge_min=cfg.hollow_rho_edge_min,
                            lam_edge=cfg.lam_strut_edge,
                            lam_void=cfg.lam_strut_void,
                            eps=cfg.eps,
                        )

                        strut_terms.append(loss_strut_i)
                        strut_edge_terms.append(loss_strut_edge_i)
                        strut_void_terms.append(loss_strut_void_i)

                uv_anchor_list = uv_anchor_next_list

                # ----------------------------------------------------
                # Aggregate face outputs
                # ----------------------------------------------------
                if active_face_count > 0:
                    participating_count_mean = participating_count_total / active_face_count
                    participating_frac_mean = participating_frac_sum / active_face_count
                    inactive_count_mean = inactive_count_total / active_face_count
                    inactive_frac_mean = inactive_frac_sum / active_face_count
                    active_weight_min_global = min(active_weight_min_list)
                    active_weight_mean_global = active_weight_mean_sum / active_face_count
                    active_weight_max_global = max(active_weight_max_list)
                else:
                    participating_count_mean = 0.0
                    participating_frac_mean = 0.0
                    inactive_count_mean = 0.0
                    inactive_frac_mean = 0.0
                    active_weight_min_global = 0.0
                    active_weight_mean_global = 0.0
                    active_weight_max_global = 0.0

                rho = rho_acc / rho_wgt.clamp_min(cfg.eps)
                rho_boundary = rho_b_acc / rho_b_wgt.clamp_min(cfg.eps)
                rho_v_all = rho_v_acc / rho_v_wgt.clamp_min(cfg.eps)
                rho_s_all = rho_s_acc / rho_s_wgt.clamp_min(cfg.eps)

                fiber_surface = fiber_acc / fiber_wgt.clamp_min(cfg.eps)[:, None]
                fiber_norm = fiber_surface.norm(dim=1, keepdim=True).clamp_min(cfg.eps)
                fiber_surface = fiber_surface / fiber_norm

                zero = torch.zeros((), dtype=dtype, device=device)

                loss_rep = self._safe_weighted_mean(rep_terms, face_weights_this_step, dtype, device, cfg.eps) if compute_rep_loss else zero
                loss_bnd = self._safe_weighted_mean(bnd_terms, face_weights_this_step, dtype, device, cfg.eps) if compute_bnd_loss else zero
                loss_strut = self._safe_weighted_mean(strut_terms, face_weights_this_step, dtype, device, cfg.eps) if compute_strut_loss else zero
                loss_strut_edge = self._safe_weighted_mean(strut_edge_terms, face_weights_this_step, dtype, device, cfg.eps) if compute_strut_loss else zero
                loss_strut_void = self._safe_weighted_mean(strut_void_terms, face_weights_this_step, dtype, device, cfg.eps) if compute_strut_loss else zero
                loss_seed_active = self._safe_weighted_mean(seed_active_terms, face_weights_this_step, dtype, device, cfg.eps) if compute_seed_active_loss else zero

                w_geo_mean = self._safe_weighted_mean(w_geo_terms, face_weights_this_step, dtype, device, cfg.eps)
                h_mean = self._safe_weighted_mean(h_terms, face_weights_this_step, dtype, device, cfg.eps)

                boundary_width_mean = self._safe_weighted_mean(boundary_width_terms, face_weights_this_step, dtype, device, cfg.eps)
                boundary_alpha_mean = self._safe_weighted_mean(boundary_alpha_terms, face_weights_this_step, dtype, device, cfg.eps)
                boundary_beta_mean = self._safe_weighted_mean(boundary_beta_terms, face_weights_this_step, dtype, device, cfg.eps)

                theta_mean = self._safe_weighted_mean(theta_mean_terms, face_weights_this_step, dtype, device, cfg.eps)
                a_metric_mean = self._safe_weighted_mean(a_metric_terms, face_weights_this_step, dtype, device, cfg.eps)

                loss_width_active = self._safe_weighted_mean(width_active_terms, face_weights_this_step, dtype, device, cfg.eps) if compute_width_active_loss else zero

                # ----------------------------------------------------
                # Volume loss
                # ----------------------------------------------------
                vol_frac_total = (rho * A_v).sum() / (A_v.sum() + cfg.eps)
                vol_frac_v = (rho_v_all * A_v).sum() / (A_v.sum() + cfg.eps)
                vol_frac_eff = vol_frac_v
                sharp_vol_ramp = 0.0
                loss_vol_sharp = zero
                loss_vol = zero

                if compute_vol_loss:
                    loss_vol_v = self.loss_volume(
                        rho=rho_v_all,
                        A_v=A_v,
                        target_volfrac=cfg.target_volfrac,
                        eps=cfg.eps,
                    )
                    loss_vol_total = self.loss_volume(
                        rho=rho,
                        A_v=A_v,
                        target_volfrac=cfg.target_volfrac,
                        eps=cfg.eps,
                    )

                    loss_vol_eff_v, vol_frac_eff = self.loss_volume.powered(
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
                    loss_vol_sharp, vol_frac_sharp = self.loss_volume(
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
                        loss_vol_base, vol_frac_weighted = self.loss_volume.with_boundary_discount(
                            rho=rho,
                            A_v=A_v,
                            rho_boundary=rho_boundary,
                            target_volfrac=cfg.target_volfrac,
                            boundary_weight=cfg.boundary_vol_weight,
                            eps=cfg.eps,
                        )
                        loss_vol_eff_weighted, vol_frac_eff = self.loss_volume.powered(
                            rho=rho,
                            A_v=(1.0 - rho_boundary + cfg.boundary_vol_weight * rho_boundary) * A_v,
                            target_volfrac=cfg.target_volfrac,
                            power=cfg.effective_volume_power,
                            eps=cfg.eps,
                        )
                        sharp_weights = (1.0 - rho_boundary + cfg.boundary_vol_weight * rho_boundary) * A_v
                        loss_vol_sharp, vol_frac_sharp = self.loss_volume(
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
                else:
                    vol_frac_sharp = (rho_s_all * A_v).sum() / (A_v.sum() + cfg.eps)
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
                    fem_out = self.loss_fem.evaluate(
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
                lam_width_active_eff = cfg.lam_width_active * self.ramp_weight(
                    step=step,
                    total_steps=cfg.num_steps,
                    start_frac=cfg.width_warmup_start_frac,
                    ramp_frac=cfg.width_warmup_ramp_frac,
                )
                if use_hard_refine_step:
                    lam_width_active_eff = lam_width_active_eff * float(cfg.hard_refine_width_multiplier)

                L_total = (
                    cfg.lam_vol * (loss_vol / n_vol)
                    + cfg.lam_rep * (loss_rep / n_rep)
                    + cfg.lam_bnd * (loss_bnd / n_bnd)
                )

                if cfg.lam_strut != 0.0:
                    L_total = L_total + cfg.lam_strut * (loss_strut / n_strut)

                if lam_width_active_eff != 0.0:
                    L_total = L_total + lam_width_active_eff * loss_width_active

                if cfg.lam_seed_active != 0.0:
                    L_total = L_total + cfg.lam_seed_active * loss_seed_active

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

                    bad_grad_info = self._nonfinite_grad_info(ppnets)
                    if bad_grad_info:
                        bad_grad_desc = ", ".join(
                            f"face={mi}:{pn}" for mi, pn in bad_grad_info[:8]
                        )
                        tqdm.write(
                            f"[step {step}] Non-finite gradients detected, optimizer step skipped. "
                            f"Examples: {bad_grad_desc}"
                        )
                        for _mi, _pn, p in self._named_trainable_params(ppnets):
                            if p.grad is not None:
                                p.grad = None
                    else:
                        pre_step_snapshot = {
                            p: p.detach().clone()
                            for _mi, _pn, p in self._named_trainable_params(ppnets)
                        }

                        grad_clip_norm = getattr(cfg, "grad_clip_norm", None)
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            params = []
                            for ppnet in ppnets:
                                params.extend([p for p in ppnet.parameters() if p.requires_grad])
                            if params:
                                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)

                        bad_grad_info = self._nonfinite_grad_info(ppnets)
                        if bad_grad_info:
                            bad_grad_desc = ", ".join(
                                f"face={mi}:{pn}" for mi, pn in bad_grad_info[:8]
                            )
                            tqdm.write(
                                f"[step {step}] Non-finite gradients remained after clipping, "
                                f"optimizer step skipped. Examples: {bad_grad_desc}"
                            )
                            for _mi, _pn, p in self._named_trainable_params(ppnets):
                                if p.grad is not None:
                                    p.grad = None
                        else:
                            if use_hard_refine_step:
                                for ppnet in ppnets:
                                    tau_head = getattr(ppnet, "tau_head", None)
                                    if cfg.freeze_tau_head_during_hard_refine and tau_head is not None:
                                        for p in tau_head.parameters():
                                            p.grad = None
                            opt.step()

                            bad_param_info = self._nonfinite_param_info(ppnets)
                            if bad_param_info:
                                bad_param_desc = ", ".join(
                                    f"face={mi}:{pn}" for mi, pn in bad_param_info[:8]
                                )
                                self._restore_param_snapshot(pre_step_snapshot)
                                bad_param_set = set(bad_param_info)
                                affected_params = [
                                    p for mi, pn, p in self._named_trainable_params(ppnets)
                                    if (mi, pn) in bad_param_set
                                ]
                                self._clear_optimizer_state_for_params(opt, affected_params)
                                for _mi, _pn, p in self._named_trainable_params(ppnets):
                                    if p.grad is not None:
                                        p.grad = None
                                tqdm.write(
                                    f"[step {step}] Non-finite parameters after opt.step(); restored previous "
                                    f"parameters and cleared optimizer state. Examples: {bad_param_desc}"
                                )
                            elif scheduler is not None:
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
                    min_seed_dist = self.min_pairwise_seed_distance(seeds_list)

                    score = float(L_total.detach().item()) if total_is_finite else float("inf")
                    if not (total_is_finite and fem_is_valid):
                        score = float("inf")

                    best_candidate_is_valid = (
                        ((cfg.lam_fem == 0.0) or fem_is_valid)
                        and total_is_finite
                        and participating_count_total >= float(cfg.min_active_seeds or 1)
                    )

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
                        best_active_count = float(participating_count_total)
                        best_inactive_count = float(inactive_count_total)
                        best_rho = rho.detach().clone()
                        best_fiber_surface = fiber_surface.detach().clone()
                        best_seeds = [s.detach().clone() for s in seeds_list]
                        best_pred = self._clone_pred_list(pred_list)

                        if improvement_gap is None or improvement_gap > 50:
                            tqdm.write(
                                f"New best_step={best_step} | "
                                f"best_score={best_score:.6f} | "
                                f"best_active_count={best_active_count:.1f} | "
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
                        "loss_seed_active": self._finite_or_default(loss_seed_active),
                        "comp": self._finite_or_default(comp_val),
                        "vol_frac": float(vol_frac.detach().item()),
                        "vol_frac_internal": float(vol_frac_v.detach().item()),
                        "vol_frac_eff": float(vol_frac_eff.detach().item()),
                        "vol_frac_sharp": float(vol_frac_sharp.detach().item()),
                        "vol_dev": float(vol_dev.detach().item()),
                        "vol_dev_eff": float(vol_dev_eff.detach().item()),
                        "tau": float(tau_step),
                        "seed_offset_scale": float(seed_offset_scale_step),
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
                        "min_seed_dist": min_seed_dist,
                        "grad_mean": g_mean,
                        "best_score": best_score,
                        "best_step": best_step,
                        "best_hard_score": self._finite_or_default(best_hard_score),
                        "best_hard_step": float(best_hard_step),
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

                        "active_count_total": participating_count_total,
                        "active_count_mean": participating_count_mean,
                        "active_frac_mean": participating_frac_mean,
                        "inactive_count_total": inactive_count_total,
                        "inactive_count_mean": inactive_count_mean,
                        "inactive_frac_mean": inactive_frac_mean,
                        "seed_active_weight_min": active_weight_min_global,
                        "seed_active_weight_mean": active_weight_mean_global,
                        "seed_active_weight_max": active_weight_max_global,
                        "hard_refine_on": 1.0 if use_hard_refine_step else 0.0,
                        "loss_width_active": self._finite_or_default(loss_width_active),
                        "lam_width_active_eff": lam_width_active_eff,
                        "anchor_update_allowed": 1.0 if anchor_update_allowed else 0.0,
                        "collapse_active": (
                            1.0
                            if participating_count_total < float(cfg.min_active_seeds or 1)
                            else 0.0
                        ),
                    }
                    history.append(row)

                    pbar.set_postfix(
                        loss=f"{row['L_total']:.3e}",
                        vol=f"{row['vol_frac_eff']:.3f}",
                        comp=f"{row['comp']:.2e}",
                        tau=f"{row['tau']:.2e}",
                        w=f"{row['w_geo_mean']:.3e}",
                        dmin=f"{row['min_seed_dist']:.3e}",
                        bw=f"{row['boundary_width_mean']:.3e}",
                        active=f"{participating_count_mean:.1f}",
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
                            loading_img=self.timelapse_loading_img,
                        )

                        loss_dict = {
                            "L_Total": row["L_total"],
                            "L_Volume": row["loss_vol"],
                            "L_FEM": row["loss_fem"],
                            "L_Active": row["loss_seed_active"],
                            "L_St": row["loss_strut"],
                            "L_Bnd": row["loss_bnd"],
                            "L_Rep": row["loss_rep"],
                        }

                        recorder.add_frame(
                            step=step,
                            cad_img=cad_img,
                            loss_dict=loss_dict,
                            title_text=(
                                f"vol_total={row['vol_frac']:.4f} | "
                                f"vol_internal={row['vol_frac_internal']:.4f} | "
                                f"vol_eff={row['vol_frac_eff']:.4f} | "
                                f"W={row['w_geo_mean']:.4g} | "
                                f"tau={row['tau']:.4g} | "
                                f"bw={row['boundary_width_mean']:.4g} | "
                                f"ba={row['boundary_alpha_mean']:.4g} | "
                                f"bb={row['boundary_beta_mean']:.4g} | "
                                f"act={row['active_count_total']:.0f} inact={row['inactive_count_total']:.0f} | "
                                f"Δrho={drho:.2e} Δseed={dseed:.2e} "
                                f"dmin={min_seed_dist:.2e} grad_mean={g_mean:.2e} | "
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
                            f"Active Seeds/Total={participating_count_total:.0f}/{participating_count_total+inactive_count_total:.0f} | "
                            f"L_total={row['L_total']:.4e} | "
                            f"L_vol={row['loss_vol']:.3e} "
                            f"L_fem={row['loss_fem']:.3e} "
                            f"L_wact={row['loss_width_active']:.3e} "
                            f"L_active={row['loss_seed_active']:.3e} "
                            f"L_strut={row['loss_strut']:.3e} "
                            f"L_rep={row['loss_rep']:.3e} "
                            f"L_bnd={row['loss_bnd']:.3e} |"
                            f"vol={row['vol_frac']:.3f} "
                            f"vol_eff={row['vol_frac_eff']:.3f} "
                            f"(/{cfg.target_volfrac:.3f}) "
                            f"tau={row['tau']:.3e} "
                            f"os={row['seed_offset_scale']:.2e} "
                            f"comp={row['comp']:.3e} | "
                            f"hard_refine={'ON' if use_hard_refine_step else 'off'} | "
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
                            f"seed_active_w(min/mean/max)={active_weight_min_global:.3f}/{active_weight_mean_global:.3f}/{active_weight_max_global:.3f} | "
                            f"Δrho={drho:.2e} Δseed={dseed:.2e} "
                            f"dmin={min_seed_dist:.2e} grad_mean={g_mean:.2e} | "
                            f"fem={fem_status} | "
                            f"best={best_score:.4e}@{best_step} | "
                            f"best_hard={best_hard_score:.4e}@{best_hard_step}"
                        )

                    rep_value = float(row["loss_rep"])
                    bnd_value = float(row["loss_bnd"])
                    vol_eff_value = float(row["vol_frac_eff"])
                    w_geo_value = float(row["w_geo_mean"])
                    min_seed_dist_value = float(row["min_seed_dist"])
                    min_seed_dist_limit = float(cfg.anchor_guard_min_seed_dist_factor) * float(cfg.w_min)

                    anchor_update_allowed = (
                        rep_value <= float(cfg.anchor_guard_rep_max)
                        and bnd_value <= float(cfg.anchor_guard_bnd_max)
                        and vol_eff_value >= float(cfg.anchor_guard_vol_eff_min)
                        and w_geo_value >= float(cfg.anchor_guard_width_factor_min) * float(cfg.w_min)
                        and min_seed_dist_value >= min_seed_dist_limit
                    )

                    if step >= cfg.early_stop_start and steps_since_improve >= cfg.patience:
                        tqdm.write(
                            f"Early stopping at step {step} | "
                            f"best_step={best_step} | best_score={best_score:.6f} |"
                        )
                        break

        # ------------------------------------------------------------
        # Fallback best state
        # ------------------------------------------------------------
        if best_rho is None:
            with torch.no_grad():
                best_rho = rho.detach().clone()
                best_seeds = [s.detach().clone() for s in seeds_list]
                best_pred = self._clone_pred_list(pred_list)
                best_step = step
                best_score = float("inf") if not self._scalar_tensor_is_finite(L_total) else float(L_total.detach().item())

                if best_vol_frac is None:
                    best_vol_frac = float(vol_frac_eff.detach().item())
                if best_comp is None:
                    best_comp = float(comp_val.detach().item())
                if best_w_geo is None:
                    best_w_geo = float(w_geo_mean.detach().item())
                if best_active_count is None:
                    best_active_count = float(participating_count_total)
                if best_inactive_count is None:
                    best_inactive_count = float(inactive_count_total)

        use_hard_result = False
        if use_hard_result:
            best_score = best_hard_score
            best_step = best_hard_step
            best_vol_frac = best_hard_vol_frac
            best_comp = best_hard_comp
            best_w_geo = best_hard_w_geo
            best_active_count = best_hard_active_count
            best_inactive_count = best_hard_inactive_count
            best_rho = best_hard_rho
            best_fiber_surface = best_hard_fiber_surface
            best_seeds = best_hard_seeds
            best_pred = best_hard_pred

        # ------------------------------------------------------------
        # Final outputs
        # ------------------------------------------------------------
        with torch.no_grad():
            hard_rho_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
            hard_rho_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)
            hard_fiber_acc = torch.zeros((vertices_number, 3), dtype=dtype, device=device)
            hard_fiber_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)
            pred_by_face_id = {p["face_id"]: p for p in best_pred} if best_pred else {}

            for ft, decoder, A_local in zip(face_tensors, decoders, local_vertex_areas):
                pred_i = pred_by_face_id.get(ft["face_id"], None)
                if pred_i is None:
                    continue

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
                seed_domain_mask_i = self._seed_domain_mask_for_face(ft)

                tau_i = self._fallback_tau_value() if pred_i.get("tau") is None else pred_i["tau"]
                hard_out_i = decoder.evaluate_at_uv(
                    points_uv=ft["uv"],
                    Xu=ft["Xu"],
                    Xv=ft["Xv"],
                    tau=tau_i,
                    seeds_raw=pred_i["seeds_raw"],
                    w_raw=pred_i["w_raw"],
                    h_raw=pred_i.get("h_raw", None),
                    theta=pred_i.get("theta", None),
                    a_raw=pred_i.get("a_raw", None),
                    points_face_id=local_face_id,
                    boundary_uv=boundary_uv_i,
                    boundary_face_id=boundary_face_id_i,
                    boundary_width_raw=pred_i.get("boundary_width_raw", None),
                    boundary_alpha_raw=pred_i.get("boundary_alpha_raw", None),
                    boundary_beta_raw=pred_i.get("boundary_beta_raw", None),
                    hard_seed_mask=True,
                    seed_domain_mask=seed_domain_mask_i,
                    seed_domain_mask_threshold=cfg.seed_domain_mask_threshold,
                    seed_domain_temp=cfg.seed_domain_temp,
                )

                gidx = ft["global_vertex_idx"]
                w_local = A_local.clamp_min(cfg.eps)
                hard_rho_acc[gidx] += hard_out_i["rho"] * w_local
                hard_rho_wgt[gidx] += w_local
                hard_fiber_acc[gidx] += hard_out_i["fiber3d"] * w_local[:, None]
                hard_fiber_wgt[gidx] += w_local

            final_shape_density = hard_rho_acc / hard_rho_wgt.clamp_min(cfg.eps)
            final_shape_fiber_direction = hard_fiber_acc / hard_fiber_wgt.clamp_min(cfg.eps)[:, None]
            final_fiber_norm = final_shape_fiber_direction.norm(dim=1, keepdim=True)
            final_shape_fiber_direction = torch.where(
                final_fiber_norm > cfg.eps,
                final_shape_fiber_direction / final_fiber_norm.clamp_min(cfg.eps),
                torch.zeros_like(final_shape_fiber_direction),
            )
            seed_points_final = self._seed_points_xyz_all_faces(best_seeds, face_tensors)

            if mid_shape_density is None:
                mid_shape_density = final_shape_density.clone()
                seed_points_mid = seed_points_final

        computation_time_sec = time.perf_counter() - train_start_time

        tqdm.write(
            f"FINAL RETURNED: best_step={best_step}, best_score={best_score:.6f} | "
            f"vol_eff={best_vol_frac:.3e}, comp={best_comp:.3e}, w_geo={best_w_geo:.3e} | "
            f"active={float(best_active_count or 0.0):.0f}, inactive={float(best_inactive_count or 0.0):.0f} | "
            f"source={'hard' if use_hard_result else 'global'} | "
            f"time={self._format_elapsed_time(computation_time_sec)}"
        )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        best_row = None
        if history and best_step >= 0:
            for row in reversed(history):
                if int(row["step"]) == int(best_step):
                    best_row = row
                    break

        if cfg.MakeTimelaps:
            try:
                total_seed_slots = int(cfg.seed_number) * max(len(best_pred) if best_pred else len(face_tensors), 1)
                active_seed_count = int(round(float(best_active_count or 0.0)))
                best_vol_total = (
                    float(best_row["vol_frac"])
                    if best_row is not None and "vol_frac" in best_row
                    else float(best_vol_frac)
                )
                best_vol_internal = (
                    float(best_row["vol_frac_internal"])
                    if best_row is not None and "vol_frac_internal" in best_row
                    else float(best_vol_frac)
                )
                best_vol_eff = float(best_vol_frac)
                tuned_param_summary = {
                    "best_step": f"{int(best_step)}",
                    "active_seeds": f"{active_seed_count}/{total_seed_slots}",
                    "Vol_total": f"{best_vol_total:.6g}",
                    "Vol_internal": f"{best_vol_internal:.6g}",
                    "Vol_eff": f"{best_vol_eff:.6g}",
                    "w": f"{float(best_w_geo):.6g}",
                    "tau": f"{self._fallback_tau_value():.6g}",
                    "h": "nan",
                    "bw": "nan",
                    "ba": "nan",
                    "bb": "nan",
                    "theta": "nan",
                    "a": "nan",
                }
                if best_pred:
                    def _mean_from_best_pred(key):
                        vals = []
                        for p in best_pred:
                            v = p.get(key)
                            if isinstance(v, torch.Tensor):
                                vals.append(float(v.detach().mean().item()))
                        if vals:
                            return float(sum(vals) / len(vals))
                        return float("nan")

                    tau_mean = _mean_from_best_pred("tau")
                    h_mean = _mean_from_best_pred("h")
                    bw_mean = _mean_from_best_pred("boundary_width")
                    ba_mean = _mean_from_best_pred("boundary_alpha")
                    bb_mean = _mean_from_best_pred("boundary_beta")
                    theta_mean_best = _mean_from_best_pred("theta_mean")
                    a_mean_best = _mean_from_best_pred("a_metric")

                    tuned_param_summary = {
                        "best_step": f"{int(best_step)}",
                        "active_seeds": f"{active_seed_count}/{total_seed_slots}",
                        "w": f"{float(best_w_geo):.6g}",
                        "tau": (
                            f"{(tau_mean if math.isfinite(tau_mean) else self._fallback_tau_value()):.6g}"
                        ),
                        "h": f"{h_mean:.6g}" if math.isfinite(h_mean) else "nan",
                        "bw": f"{bw_mean:.6g}" if math.isfinite(bw_mean) else "nan",
                        "ba": f"{ba_mean:.6g}" if math.isfinite(ba_mean) else "nan",
                        "bb": f"{bb_mean:.6g}" if math.isfinite(bb_mean) else "nan",
                        "theta": f"{theta_mean_best:.6g}" if math.isfinite(theta_mean_best) else "nan",
                        "a": f"{a_mean_best:.6g}" if math.isfinite(a_mean_best) else "nan",
                    }

                best_loss_dict = {
                    "L_Total": float(best_score),
                    "L_Volume": float(best_row["loss_vol"]) if best_row is not None else float("nan"),
                    "L_FEM": float(best_row["loss_fem"]) if best_row is not None else float("nan"),
                    "L_Active": float(best_row["loss_seed_active"]) if best_row is not None else float("nan"),
                    "L_Strut": float(best_row["loss_strut"]) if best_row is not None else float("nan"),
                    "L_Bnd": float(best_row["loss_bnd"]) if best_row is not None else float("nan"),
                    "L_Rep": float(best_row["loss_rep"]) if best_row is not None else float("nan"),
                }
                results_text = (
                    f"vol_total={best_vol_total:.6g} | "
                    f"vol_internal={best_vol_internal:.6g} | "
                    f"vol_eff={best_vol_eff:.6g} | "
                    f"fem={float(best_comp):.6g} | "
                    f"compute_time={self._format_elapsed_time(computation_time_sec)}"
                )
                tuned_param_title = " | ".join(f"{key}={value}" for key, value in tuned_param_summary.items())

                best_cad_img = self._render_current_cad_frame_cached(
                    seeds_list=best_seeds,
                    decoders=decoders,
                    pred_list=best_pred,
                    render_cache=render_cache,
                    thr=getattr(cfg, "vis_thr", cfg.TM_laps_Thr),
                    loading_img=self.timelapse_loading_img,
                )
                best_frame_path = recorder.add_frame(
                    step=cfg.num_steps + 1,
                    cad_img=best_cad_img,
                    loss_dict=best_loss_dict,
                    title_text=tuned_param_title,
                    highlight_best=True,
                    chart_title="Best Result Losses",
                    summary_title="Tuned Parameters",
                    prefix_step_in_summary=False,
                    results_title="Results",
                    results_text=results_text,
                )
                if timelapse_output_folder:
                    shutil.copy2(
                        best_frame_path,
                        os.path.join(timelapse_output_folder, "best_result_frame.png"),
                    )
                recorder.build_video(hold_last_seconds=10.0)
            except Exception as e:
                tqdm.write(f"Failed to build timelapse video: {e}")

        return {
            "decoders": decoders,
            "ppnets": ppnets,
            "optimizer": opt,
            "history": history,
            "best_score": best_score,
            "best_step": best_step,
            "best_active_count": float(best_active_count or 0.0),
            "best_inactive_count": float(best_inactive_count or 0.0),
            "best_hard_score": best_hard_score,
            "best_hard_step": best_hard_step,
            "best_hard_active_count": float(best_hard_active_count or 0.0),
            "best_hard_inactive_count": float(best_hard_inactive_count or 0.0),
            "returned_best_source": "hard" if use_hard_result else "global",
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
            "uv_anchor_list": uv_anchor_list,
            "face_tensors": face_tensors,
            "fem_debug_history": self.fem_debug_history,
            "last_fem_debug": self.last_fem_debug,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "shape_path": shape_path,
        }
