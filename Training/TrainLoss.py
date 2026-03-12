import torch


class TrainLoss:
    """
    Loss terms for density/seed optimization with FEM-based objectives.

    FEM conventions
    ---------------
    stress -> Fmin from FEM solver (larger is better)
    comp   -> compliance          (smaller is better)
    """

    def __init__(self, fem, shell_problem):
        self.fem = fem
        self.shell_problem = shell_problem
        self.last_fem_debug = {}
        self.fem_debug_history = []

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
        gates: torch.Tensor | None = None,
        margin: float = 0.05,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        u, v = seeds[:, 0], seeds[:, 1]
        d = torch.stack([u, 1 - u, v, 1 - v], dim=1)
        penalty = torch.exp(-d / (margin + eps)).mean(dim=1)

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
    def stress_loss(
        stress: torch.Tensor,
        mode: str = "log",
        target_min: float | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        stress = stress.reshape(())

        if mode == "inverse":
            return 1.0 / (stress + eps)

        if mode == "log":
            return -torch.log(stress + eps)

        if mode == "hinge":
            if target_min is None:
                raise ValueError("target_min must be provided when mode='hinge'")
            target = torch.as_tensor(target_min, dtype=stress.dtype, device=stress.device)
            return torch.relu(target - stress) ** 2

        raise ValueError(f"Unknown stress loss mode: {mode}")

    def fem_loss(
        self,
        rho_surface: torch.Tensor,
        fiber_surface: torch.Tensor,
        w_comp: float = 1.0,
        w_stress: float = 1.0,
        comp_normalize_by: float | None = None,
        stress_mode: str = "log",
        stress_target_min: float | None = None,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> dict:
        """
        Build FEM fields from decoder outputs and compute FEM-based losses.
        Diagnostics are stored in self.last_fem_debug, never printed.
        """
        device = rho_surface.device
        dtype = rho_surface.dtype

        fem_fields = self.shell_problem.build_fem_fields_from_decoder(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
        )

        density = torch.as_tensor(fem_fields["density"], dtype=dtype, device=device)
        phi = torch.as_tensor(fem_fields["phi"], dtype=dtype, device=device)
        theta = torch.as_tensor(fem_fields["theta"], dtype=dtype, device=device)

        fiber_norm = torch.linalg.norm(fiber_surface, dim=1)

        debug = {
            "rho_surface_shape": tuple(rho_surface.shape),
            "fiber_surface_shape": tuple(fiber_surface.shape),
            "density_shape": tuple(density.shape),
            "phi_shape": tuple(phi.shape),
            "theta_shape": tuple(theta.shape),

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

            "void_fraction_lt_1e_2": float((density < 1e-2).float().mean().detach().item()),
            "void_fraction_lt_5e_2": float((density < 5e-2).float().mean().detach().item()),
        }

        stress, comp = self.fem(density, phi, theta, penal=3)

        debug.update({
            "stress_is_finite": bool(torch.isfinite(stress).detach().item()),
            "comp_is_finite": bool(torch.isfinite(comp).detach().item()),
            "stress_value": float(torch.nan_to_num(stress, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
            "comp_value": float(torch.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
        })

        loss_comp = self.compliance_loss(
            comp=comp,
            normalize_by=comp_normalize_by,
            eps=eps,
        )

        loss_stress = self.stress_loss(
            stress=stress,
            mode=stress_mode,
            target_min=stress_target_min,
            eps=eps,
        )

        fem_total = w_comp * loss_comp + w_stress * loss_stress

        debug.update({
            "loss_comp_is_finite": bool(torch.isfinite(loss_comp).detach().item()),
            "loss_stress_is_finite": bool(torch.isfinite(loss_stress).detach().item()),
            "fem_total_is_finite": bool(torch.isfinite(fem_total).detach().item()),
            "loss_comp_value": float(torch.nan_to_num(loss_comp, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
            "loss_stress_value": float(torch.nan_to_num(loss_stress, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
            "fem_total_value": float(torch.nan_to_num(fem_total, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
        })

        self.last_fem_debug = debug
        if save_debug_history:
            self.fem_debug_history.append(debug.copy())

        return {
            "fem_total": fem_total,
            "stress": stress,
            "comp": comp,
            "compliance_loss": loss_comp,
            "stress_loss": loss_stress,
        }

    def total_loss(
        self,
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        seeds: torch.Tensor,
        fiber_surface: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
        w_vol: float = 1.0,
        w_seed: float = 1.0,
        w_boundary: float = 1.0,
        w_fem: float = 0.0,
        w_comp: float = 1.0,
        w_stress: float = 1.0,
        sigma: float = 0.08,
        margin: float = 0.05,
        comp_normalize_by: float | None = None,
        stress_mode: str = "log",
        stress_target_min: float | None = None,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> dict:
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
            "stress": torch.zeros((), dtype=rho.dtype, device=rho.device),
            "comp": torch.zeros((), dtype=rho.dtype, device=rho.device),
            "compliance_loss": torch.zeros((), dtype=rho.dtype, device=rho.device),
            "stress_loss": torch.zeros((), dtype=rho.dtype, device=rho.device),
        }

        if w_fem != 0.0:
            if fiber_surface is None:
                raise ValueError("fiber_surface must be provided when w_fem != 0")

            fem_out = self.fem_loss(
                rho_surface=rho,
                fiber_surface=fiber_surface,
                w_comp=w_comp,
                w_stress=w_stress,
                comp_normalize_by=comp_normalize_by,
                stress_mode=stress_mode,
                stress_target_min=stress_target_min,
                eps=eps,
                save_debug_history=save_debug_history,
            )
            total = total + w_fem * fem_out["fem_total"]

        return {
            "total": total,
            "volume": loss_vol,
            "seed_repulsion": loss_seed,
            "boundary_repulsion": loss_boundary,
            "fem_total": fem_out["fem_total"],
            "stress": fem_out["stress"],
            "comp": fem_out["comp"],
            "compliance_loss": fem_out["compliance_loss"],
            "stress_loss": fem_out["stress_loss"],
        }