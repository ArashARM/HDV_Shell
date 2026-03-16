from dataclasses import dataclass
import math
import torch


@dataclass
class TrainingConfig:
    seed_number: int = 15
    use_anisotropy: bool = True
    fixed_height: float | None = None
    target_volfrac: float = 0.5
    seed_repulsion_sigma: float = 0.08
    boundary_margin: float = 0.05


    # only 4 outer weights
    lam_fem: float = 1.0
    lam_vol: float = 2.0
    lam_rep: float = 0.5
    lam_bnd: float = 0.5

    # best model selection
    lam_best_vol: float = 5.0
    lam_best_fem: float = 0.0

    # FEM scaling
    comp_normalize_by: float | None = 1e10

    # normalize losses online
    normalize_losses: bool = True

    # FEM robustness
    fem_density_floor: float = 0.02
    skip_bad_fem_steps: bool = True

    num_steps: int = 10000
    context_vector_size: int = 8

    tau: float = 0.02
    beta: float = 0.05

    lr_seed_refine: float = 2e-4
    lr_delta_head: float = 2e-4
    lr_mlp: float = 2e-4
    lr_w_head: float = 2e-4
    lr_h_head: float = 2e-4

    log_every: int = 50
    early_stop_start: int = 300
    patience: int = 300
    min_delta: float = 1e-4

    eps: float = 1e-12


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

    # ------------------------------------------------------------------
    # Loss utilities
    # ------------------------------------------------------------------

    @staticmethod
    def volume_loss_constant_height(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        vol_frac = (rho * A_v).sum() / (A_v.sum() + eps)
        return (vol_frac - target_volfrac) ** 2

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
        """
        Penalize seeds that get too close to the *actual trimmed face boundary*.

        boundary_uv:
            [Nb, 2] UV samples from the triangulated face boundary.
            If the face has no open boundary (e.g. effectively closed in the
            optimization domain), return zero penalty.
        """
        if boundary_uv is None or boundary_uv.numel() == 0:
            return torch.zeros((), dtype=seeds.dtype, device=seeds.device)

        # Distance from each seed to the nearest boundary sample
        dmin = torch.cdist(seeds, boundary_uv).amin(dim=1)

        # Smooth penalty: large near boundary, decays away from it
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
    def _scalar_tensor_is_finite(x: torch.Tensor) -> bool:
        return bool(torch.isfinite(x).reshape(()).detach().item())

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
        w_fem: float = 0.0,
        comp_normalize_by: float | None = None,
        density_floor: float = 0.02,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> dict:
        
        sigma= self.cfg.seed_repulsion_sigma
        margin= self.cfg.boundary_margin
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

        total = (
            w_vol * loss_vol
            + w_seed * loss_seed
            + w_boundary * loss_boundary
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
        use_anisotropy,
        context_vector_size,
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
            use_anisotropy=use_anisotropy,
            fixed_height=getattr(self.cfg.fixed_height, "thickness", None),
        ).to(device)

        ppnet = self.ppnet_cls(
            context_dim=context_vector_size,
            n_seeds=seed_number,
            use_anisotropy=use_anisotropy,
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
                use_anisotropy=cfg.use_anisotropy,
                context_vector_size=cfg.context_vector_size,
            )
            decoders.append(decoder)
            ppnets.append(ppnet)

        return decoders, ppnets

    def _build_optimizer(self, ppnets):
        cfg = self.cfg
        param_groups = []

        for ppnet in ppnets:
            param_groups.extend([
                {"params": ppnet.seed_refine.parameters(), "lr": cfg.lr_seed_refine},
                {"params": ppnet.delta_head.parameters(),  "lr": cfg.lr_delta_head},
                {"params": ppnet.mlp.parameters(),         "lr": cfg.lr_mlp},
                {"params": ppnet.w_head.parameters(),      "lr": cfg.lr_w_head},
            ])

            fixed_height = getattr(self.cfg.fixed_height, "thickness", None)
            if (fixed_height is None) and hasattr(ppnet, "h_head"):
                param_groups.append({"params": ppnet.h_head.parameters(), "lr": cfg.lr_h_head})

            if cfg.use_anisotropy:
                if hasattr(ppnet, "theta_head"):
                    param_groups.append({"params": ppnet.theta_head.parameters(), "lr": cfg.lr_mlp})
                if hasattr(ppnet, "a_head"):
                    param_groups.append({"params": ppnet.a_head.parameters(), "lr": cfg.lr_mlp})

        return torch.optim.Adam(param_groups)

    def _init_face_seeds(self, face_tensors):
        cfg = self.cfg
        uv_init_list = []

        for ft in face_tensors:
            boundary = torch.unique(ft["boundary_idx_ring1"])
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

    def train(
        self,
        uv,
        Xu,
        Xv,
        points_xyz,
        face_areas,
        faces_ijk,
        face_id,
        boundary_idx_ring1,
        face_tensors=None,
    ):
        cfg = self.cfg
        device = uv.device
        dtype = uv.dtype
        mid_step = cfg.num_steps // 2
        vertices_number = uv.shape[0]

        if face_tensors is None:
            face_tensors = [{
                "face_id": 0,
                "uv": uv,
                "Xu": Xu,
                "Xv": Xv,
                "points_xyz": points_xyz,
                "faces_ijk": faces_ijk,
                "face_areas": face_areas,
                "boundary_idx_ring1": boundary_idx_ring1,
                "u_periodic": False,
                "v_periodic": False,
                "global_vertex_idx": torch.arange(vertices_number, device=device, dtype=torch.long),
            }]

        A_v = self.generator.vertex_area_lumped(vertices_number, faces_ijk, face_areas)

        decoders, ppnets = self._build_face_models(face_tensors=face_tensors, device=device)
        uv_init_list = self._init_face_seeds(face_tensors)
        contexts = [
            torch.zeros(1, cfg.context_vector_size, device=device, dtype=dtype)
            for _ in face_tensors
        ]

        opt = self._build_optimizer(ppnets)

        norm_vol = RunningNorm()
        norm_rep = RunningNorm()
        norm_bnd = RunningNorm()
        norm_fem = RunningNorm()

        best_score = float("inf")
        best_step = -1
        best_rho = None
        best_seeds = None
        best_pred = None
        steps_since_improve = 0

        initial_shape_density = None
        mid_shape_density = None
        final_shape_density = None

        seed_points_init = None
        seed_points_mid = None
        seed_points_final = None

        rho0 = None
        seeds0 = None
        history = []

        for step in range(cfg.num_steps):
            rho = torch.zeros((vertices_number,), dtype=dtype, device=device)
            fiber_surface = torch.zeros((vertices_number, 3), dtype=dtype, device=device)
            fiber_surface[:, 0] = 1.0

            seeds_list = []
            pred_list = []
            rep_terms = []
            bnd_terms = []

            for ft, decoder, ppnet, uv_init_i, context_i in zip(face_tensors, decoders, ppnets, uv_init_list, contexts):
                pred_i = ppnet(context_i, uv_init_i, offset_scale=1, clamp01=True)

                seeds_raw_i = pred_i["seeds_raw"][0]
                w_raw_i = pred_i["w_raw"][0]
                fixed_height = getattr(self.cfg.fixed_height, "thickness", None)
                h_raw_i = None if fixed_height is not None else pred_i["h_raw"][0]
                gates_i = pred_i.get("gate_probs", None)
                gates_i = gates_i[0] if gates_i is not None else None

                theta_i = pred_i["theta"][0] if (cfg.use_anisotropy and "theta" in pred_i) else None
                a_raw_i = pred_i["a_raw"][0] if (cfg.use_anisotropy and "a_raw" in pred_i) else None

                local_face_id = torch.zeros(
                    ft["uv"].shape[0],
                    dtype=torch.long,
                    device=device,
                )

                _, _, _, seeds_i, rho_i, _, _, _, _, fiber3d_i = decoder(
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
                )

                gidx = ft["global_vertex_idx"]
                rho[gidx] = rho_i
                fiber_surface[gidx] = fiber3d_i

                seeds_list.append(seeds_i)
                pred_list.append({
                    "face_id": ft["face_id"],
                    "seeds_raw": seeds_raw_i,
                    "w_raw": w_raw_i,
                    "h_raw": None if h_raw_i is None else h_raw_i,
                    "gates": None if gates_i is None else gates_i,
                })

                rep_terms.append(
                    self.seed_repulsion_term(
                        seeds=seeds_i,
                        gates=gates_i,
                        sigma=0.08,
                        eps=cfg.eps,
                    )
                )
                boundary_uv_i = None
                if ("boundary_idx" in ft) and (ft["boundary_idx"] is not None) and (ft["boundary_idx"].numel() > 0):
                    boundary_uv_i = ft["uv"][ft["boundary_idx"]]

                bnd_terms.append(
                    self.boundary_repulsion_term(
                        seeds=seeds_i,
                        boundary_uv=boundary_uv_i,
                        gates=gates_i,
                        margin=0.05,
                        eps=cfg.eps,
                    )
                )

            loss_vol = self.volume_loss_constant_height(
                rho=rho,
                A_v=A_v,
                target_volfrac=cfg.target_volfrac,
                eps=cfg.eps,
            )

            loss_rep = torch.stack(rep_terms).mean() if len(rep_terms) else torch.zeros((), dtype=dtype, device=device)
            loss_bnd = torch.stack(bnd_terms).mean() if len(bnd_terms) else torch.zeros((), dtype=dtype, device=device)

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
                    save_debug_history=True,
                )

            loss_fem = fem_out["fem_total"]
            loss_comp = fem_out["compliance_loss"]
            comp_val = fem_out["comp"]
            fem_is_valid = bool(fem_out["fem_valid"])
            fem_failure_reason = fem_out["failure_reason"]




            if cfg.normalize_losses:
                n_vol = norm_vol.update(loss_vol.detach().item())
                n_rep = norm_rep.update(loss_rep.detach().item())
                n_bnd = norm_bnd.update(loss_bnd.detach().item())
                n_fem = (
                    norm_fem.update(loss_fem.detach().item())
                    if (cfg.lam_fem != 0.0 and fem_is_valid)
                    else 1.0
                )
            else:
                n_vol = n_rep = n_bnd = n_fem = 1.0

            L_total = (
                cfg.lam_vol * (loss_vol / n_vol) +
                cfg.lam_rep * (loss_rep / n_rep) +
                cfg.lam_bnd * (loss_bnd / n_bnd)
            )

            if cfg.lam_fem != 0.0:
                if fem_is_valid:
                    L_total = L_total + cfg.lam_fem * (loss_fem / n_fem)
                elif not cfg.skip_bad_fem_steps:
                    L_total = L_total + cfg.lam_fem * loss_fem

            total_is_finite = self._scalar_tensor_is_finite(L_total)

            if total_is_finite:
                L_total.backward()
                opt.step()
            else:
                print(f"[step {step}] L_total is non-finite, optimizer step skipped.")

            with torch.no_grad():
                vol_frac = (rho * A_v).sum() / (A_v.sum() + cfg.eps)
                vol_dev = torch.abs(vol_frac - cfg.target_volfrac)

                score_fem_term = (
                    cfg.lam_best_fem * float(loss_fem.detach().item())
                    if fem_is_valid and self._scalar_tensor_is_finite(loss_fem)
                    else 0.0
                )

                # score = float(
                #     (float(L_total.detach().item()) if total_is_finite else float("inf"))
                #     + cfg.lam_best_vol * vol_dev.detach().item()
                #     + score_fem_term
                # )
                score = float(L_total.detach().item()) if total_is_finite else float("inf")

                if score < (best_score - cfg.min_delta):
                    best_score = score
                    best_step = step
                    best_rho = rho.detach().clone()
                    best_seeds = [s.detach().clone() for s in seeds_list]
                    best_pred = [
                        {
                            "face_id": p["face_id"],
                            "seeds_raw": p["seeds_raw"].detach().clone(),
                            "w_raw": p["w_raw"].detach().clone(),
                            "h_raw": None if p["h_raw"] is None else p["h_raw"].detach().clone(),
                            "gates": None if p["gates"] is None else p["gates"].detach().clone(),
                        }
                        for p in pred_list
                    ]
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1

                if step == 0:
                    initial_shape_density = rho.detach().clone()
                    seed_points_init = self._seed_points_xyz_all_faces(seeds_list, face_tensors)

                if step == mid_step:
                    mid_shape_density = rho.detach().clone()
                    seed_points_mid = self._seed_points_xyz_all_faces(seeds_list, face_tensors)

                if rho0 is None:
                    rho0 = rho.detach().clone()
                if seeds0 is None:
                    seeds0 = [s.detach().clone() for s in seeds_list]

                drho = float((rho - rho0).abs().mean().item())
                dseed_terms = [
                    float((s - s0).abs().mean().item())
                    for s, s0 in zip(seeds_list, seeds0)
                ]
                dseed = sum(dseed_terms) / max(len(dseed_terms), 1)

                rho_min = float(rho.min().item())
                rho_mean = float(rho.mean().item())
                rho_max = float(rho.max().item())

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
                    "loss_fem": self._finite_or_default(loss_fem),
                    "loss_comp": self._finite_or_default(loss_comp),
                    "comp": self._finite_or_default(comp_val),
                    "vol_frac": float(vol_frac.detach().item()),
                    "vol_dev": float(vol_dev.detach().item()),
                    "rho_min": rho_min,
                    "rho_mean": rho_mean,
                    "rho_max": rho_max,
                    "drho": drho,
                    "dseed": dseed,
                    "grad_mean": g_mean,
                    "best_score": best_score,
                    "best_step": best_step,
                    "fem_valid": fem_is_valid,
                    "fem_failure_reason": fem_failure_reason,
                    "optimizer_step_skipped": not total_is_finite,
                }
                history.append(row)

                if (not fem_is_valid) and cfg.skip_bad_fem_steps:
                    self._print_fem_failure(step)

                if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
                    fem_status = "OK" if fem_is_valid else f"BAD({fem_failure_reason})"
                    print(
                        f"[{step:05d}] "
                        f"L_total={row['L_total']:.4e} | "
                        f"L_vol={row['loss_vol']:.3e} "
                        f"L_rep={row['loss_rep']:.3e} "
                        f"L_bnd={row['loss_bnd']:.3e} "
                        f"L_fem={row['loss_fem']:.3e} |"
                        f"comp={row['comp']:.3e} "
                        f"vol={row['vol_frac']:.3f} (/{cfg.target_volfrac:.3f})"
                        f"rho(min/mean/max)={rho_min:.3f}/{rho_mean:.3f}/{rho_max:.3f} | "
                        f"Δrho={drho:.2e} Δseed={dseed:.2e} grad_mean={g_mean:.2e} | "
                        f"fem={fem_status} | "
                        f"best={best_score:.4e}@{best_step}"
                    )

                if step >= cfg.early_stop_start and steps_since_improve >= cfg.patience:
                    print(
                        f"Early stopping at step {step} | "
                        f"best_step={best_step} | best_score={best_score:.6f}"
                    )
                    break

        if best_rho is None:
            with torch.no_grad():
                best_rho = rho.detach().clone()
                best_seeds = [s.detach().clone() for s in seeds_list]
                best_step = step
                best_score = float("inf") if not self._scalar_tensor_is_finite(L_total) else float(L_total.detach().item())

        with torch.no_grad():
            final_shape_density = best_rho.clone()
            seed_points_final = self._seed_points_xyz_all_faces(best_seeds, face_tensors)

            if mid_shape_density is None:
                mid_shape_density = final_shape_density.clone()
                seed_points_mid = seed_points_final

        print(f"FINAL RETURNED: best_step={best_step}, best_score={best_score:.6f}")

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
            "seed_points_init": seed_points_init,
            "seed_points_mid": seed_points_mid,
            "seed_points_final": seed_points_final,
            "A_v": A_v,
            "uv_init_list": uv_init_list,
            "face_tensors": face_tensors,
            "fem_debug_history": self.fem_debug_history,
            "last_fem_debug": self.last_fem_debug,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_result_stepwise(self, result, points_xyz, faces_ijk):
        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        self.viz.plot_density_and_seedpoints_3stage_2(
            mesh_points=points_xyz.detach().cpu().numpy(),
            pv_faces=pv_faces_fixed,
            density_init=result["Initial_shape_density"].detach().cpu().numpy(),
            density_mid=result["Mid_shape_density"].detach().cpu().numpy(),
            density_final=result["Final_shape_density"].detach().cpu().numpy(),
            seed_points_init=result["seed_points_init"],
            seed_points_mid=result["seed_points_mid"],
            seed_points_final=result["seed_points_final"],
            thr=0.5,
            show_shell_background=True,
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