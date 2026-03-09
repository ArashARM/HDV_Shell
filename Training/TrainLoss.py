import torch


class TrainLoss:
    """
    Loss terms for density/seed optimization.
    """

    @staticmethod
    def volume_loss_constant_height(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Penalize deviation from target volume fraction.

        rho:             (N,) density
        A_v:             (N,) per-vertex area weights
        target_volfrac:  scalar target volume fraction
        """
        vol_frac = (rho * A_v).sum() / (A_v.sum() + eps)
        return (vol_frac - target_volfrac) ** 2

    @staticmethod
    def seed_repulsion_term(
        seeds: torch.Tensor,
        gates: torch.Tensor | None = None,
        sigma: float = 0.08,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Encourage seeds to stay apart in UV space.

        seeds: (S,2)
        gates: (S,) optional activation weights
        """
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
        """
        Penalize seeds close to UV boundary [0,1]^2.

        seeds: (S,2)
        gates: (S,) optional activation weights
        """
        u, v = seeds[:, 0], seeds[:, 1]
        d = torch.stack([u, 1 - u, v, 1 - v], dim=1)
        penalty = torch.exp(-d / margin).mean(dim=1)

        if gates is None:
            return penalty.mean()

        g = gates.view(-1)
        return (g * penalty).sum() / (g.sum() + eps)

    @classmethod
    def total_loss(
        cls,
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        seeds: torch.Tensor,
        gates: torch.Tensor | None = None,
        w_vol: float = 1.0,
        w_seed: float = 1.0,
        w_boundary: float = 1.0,
        sigma: float = 0.08,
        margin: float = 0.05,
        eps: float = 1e-12,
    ):
        """
        Combined loss helper.
        Returns dict with all components and total.
        """
        loss_vol = cls.volume_loss_constant_height(
            rho=rho,
            A_v=A_v,
            target_volfrac=target_volfrac,
            eps=eps,
        )

        loss_seed = cls.seed_repulsion_term(
            seeds=seeds,
            gates=gates,
            sigma=sigma,
            eps=eps,
        )

        loss_boundary = cls.boundary_repulsion_term(
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

        return {
            "total": total,
            "volume": loss_vol,
            "seed_repulsion": loss_seed,
            "boundary_repulsion": loss_boundary,
        }