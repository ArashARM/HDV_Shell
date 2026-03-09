import copy
from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    seed_number: int = 15
    use_anisotropy: bool = True
    target_volfrac: float = 0.5

    lam_fem: float = 1.0
    lam_vol: float = 2.0
    lam_rep: float = 0.5
    lam_bnd: float = 0.5
    lam_best_vol: float = 5.0

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
    sweep_every: int | None = 300
    early_stop_start: int = 300
    patience: int = 300
    min_delta: float = 1e-4

    eps: float = 1e-12


class RunningNorm:
    def __init__(self, momentum=0.99, eps=1e-12):
        self.val = None
        self.momentum = momentum
        self.eps = eps

    def update(self, x: float) -> float:
        x = abs(float(x)) + self.eps
        if self.val is None:
            self.val = x
        else:
            self.val = self.momentum * self.val + (1.0 - self.momentum) * x
        return max(self.val, 1e-8)


class NN_Trainer:
    """
    Trainer for Voronoi / density optimization.

    Expected external objects:
      - generator
      - loss_obj
      - viz
      - decoder_cls
      - ppnet_cls
    """

    def __init__(self, generator, loss_obj, viz, decoder_cls, ppnet_cls, config: TrainingConfig):
        self.generator = generator
        self.loss_obj = loss_obj
        self.viz = viz
        self.decoder_cls = decoder_cls
        self.ppnet_cls = ppnet_cls
        self.cfg = config

    def _build_models(
        self,
        device,
        seed_number,
        boundary_idx_ring1,
        face_id,
        use_anisotropy,
        context_vector_size,
    ):
        face_u_periodic = torch.tensor([False], device=device)
        face_v_periodic = torch.tensor([False], device=device)
        seed_face_id = torch.zeros(seed_number, dtype=torch.long, device=device)

        decoder = self.decoder_cls(
            n_seeds=seed_number,
            boundary_solid_idx=boundary_idx_ring1,
            seed_face_id=seed_face_id,
            face_u_periodic=face_u_periodic,
            face_v_periodic=face_v_periodic,
            use_anisotropy=False,
        ).to(device)

        ppnet = self.ppnet_cls(
            context_dim=context_vector_size,
            n_seeds=seed_number,
            use_anisotropy=use_anisotropy,
        ).to(device)

        return decoder, ppnet

    def _build_optimizer(self, ppnet):
        cfg = self.cfg
        return torch.optim.Adam([
            {"params": ppnet.seed_refine.parameters(), "lr": cfg.lr_seed_refine},
            {"params": ppnet.delta_head.parameters(),  "lr": cfg.lr_delta_head},
            {"params": ppnet.mlp.parameters(),         "lr": cfg.lr_mlp},
            {"params": ppnet.w_head.parameters(),      "lr": cfg.lr_w_head},
            {"params": ppnet.h_head.parameters(),      "lr": cfg.lr_h_head},
        ])

    def train(
        self,
        uv,
        points_xyz,
        face_areas,
        faces_ijk,
        face_id,
        boundary_idx_ring1,
    ):
        cfg = self.cfg
        device = uv.device
        dtype = uv.dtype
        mid_step = cfg.num_steps // 2
        vertices_number = uv.shape[0]

        # ---------------- boundary / init seeds ----------------
        boundary = torch.unique(boundary_idx_ring1)
        seed_idx = self.generator.fps_3d(
        points_xyz,
        cfg.seed_number,
        exclude_idx=boundary,
        )
        uv_init = uv[seed_idx].clone()

        # ---------------- constants ----------------
        A_v = self.generator.vertex_area_lumped(vertices_number, faces_ijk, face_areas)

        decoder, ppnet = self._build_models(
            device=device,
            seed_number=cfg.seed_number,
            boundary_idx_ring1=boundary_idx_ring1,
            face_id=face_id,
            use_anisotropy=cfg.use_anisotropy,
            context_vector_size=cfg.context_vector_size,
        )

        opt = self._build_optimizer(ppnet)
        context = torch.zeros(1, cfg.context_vector_size, device=device, dtype=dtype)

        # ---------------- running normalization ----------------
        norm_vol = RunningNorm()
        norm_rep = RunningNorm()
        norm_bnd = RunningNorm()

        # ---------------- tracking ----------------
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
            decoder.beta = cfg.beta

            opt.zero_grad(set_to_none=True)

            # ---- PPNet forward ----
            pred = ppnet(context, uv_init, offset_scale=0.15, clamp01=True)
            seeds_raw = pred["seeds_raw"][0]
            w_raw = pred["w_raw"][0]
            h_raw = pred["h_raw"][0]
            gates = pred.get("gate_probs", None)
            gates = gates[0] if gates is not None else None

            # ---- decoder forward ----
            w_soft, d, M, seeds, rho, t_raw, t_uv, h, Q_used = decoder(
                uv,
                tau=cfg.tau,
                seeds_raw=seeds_raw,
                w_raw=w_raw,
                h_raw=h_raw,
                points_face_id=face_id,
            )

            # ---- losses ----
            loss_vol = self.loss_obj.volume_loss_constant_height(
                rho=rho,
                A_v=A_v,
                target_volfrac=cfg.target_volfrac,
                eps=cfg.eps,
            )

            loss_rep = self.loss_obj.seed_repulsion_term(
                seeds=seeds,
                gates=gates,
                sigma=0.08,
                eps=cfg.eps,
            )

            loss_bnd = self.loss_obj.boundary_repulsion_term(
                seeds=seeds,
                gates=gates,
                margin=0.05,
                eps=cfg.eps,
            )

            # Optional normalized combination
            n_vol = norm_vol.update(loss_vol.detach().item())
            n_rep = norm_rep.update(loss_rep.detach().item())
            n_bnd = norm_bnd.update(loss_bnd.detach().item())

            L_total = (
                cfg.lam_vol * (loss_vol / n_vol) +
                cfg.lam_rep * (loss_rep / n_rep) +
                cfg.lam_bnd * (loss_bnd / n_bnd)
            )

            L_total.backward()
            opt.step()

            # ---- stats ----
            with torch.no_grad():
                vol_frac = (rho * A_v).sum() / (A_v.sum() + cfg.eps)
                vol_dev = torch.abs(vol_frac - cfg.target_volfrac)

                # selection score for "best solution"
                score = float(
                    L_total.detach().item() +
                    cfg.lam_best_vol * vol_dev.detach().item()
                )

                if score < (best_score - cfg.min_delta):
                    best_score = score
                    best_step = step
                    best_rho = rho.detach().clone()
                    best_seeds = seeds.detach().clone()
                    best_pred = {
                        "seeds_raw": seeds_raw.detach().clone(),
                        "w_raw": w_raw.detach().clone(),
                        "h_raw": h_raw.detach().clone(),
                        "gates": None if gates is None else gates.detach().clone(),
                    }
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1

                # ---- snapshots ----
                if step == 0:
                    initial_shape_density = rho.detach().clone()
                    seed_points_init = self.generator.seeds_uv_to_xyz_nearest(seeds, uv, points_xyz)

                if step == mid_step:
                    mid_shape_density = rho.detach().clone()
                    seed_points_mid = self.generator.seeds_uv_to_xyz_nearest(seeds, uv, points_xyz)

                # ---- logging baselines ----
                if rho0 is None:
                    rho0 = rho.detach().clone()
                if seeds0 is None:
                    seeds0 = seeds.detach().clone()

                drho = float((rho - rho0).abs().mean().item())
                dseed = float((seeds - seeds0).abs().mean().item())

                rho_min = float(rho.min().item())
                rho_mean = float(rho.mean().item())
                rho_max = float(rho.max().item())

                g_mean = 0.0
                g_count = 0
                for p in ppnet.parameters():
                    if p.grad is not None:
                        g_mean += float(p.grad.detach().abs().mean().item())
                        g_count += 1
                g_mean = g_mean / max(g_count, 1)

                row = {
                    "step": step,
                    "L_total": float(L_total.detach().item()),
                    "loss_vol": float(loss_vol.detach().item()),
                    "loss_rep": float(loss_rep.detach().item()),
                    "loss_bnd": float(loss_bnd.detach().item()),
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
                }
                history.append(row)

                if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
                    print(
                        f"[{step:05d}] "
                        f"L={row['L_total']:.4e} "
                        f"vol={row['vol_frac']:.3f} dev={row['vol_dev']:.3f} target={cfg.target_volfrac:.3f} "
                        f"rho(min/mean/max)={rho_min:.3f}/{rho_mean:.3f}/{rho_max:.3f} "
                        f"rep={row['loss_rep']:.3e} bnd={row['loss_bnd']:.3e} "
                        f"Δrho={drho:.2e} Δseed={dseed:.2e} "
                        f"grad_mean={g_mean:.2e} "
                        f"best={best_score:.4e}@{best_step}"
                    )

                # ---- early stopping ----
                if (
                    step >= cfg.early_stop_start and
                    steps_since_improve >= cfg.patience
                ):
                    print(f"Early stopping at step {step} | best_step={best_step} | best_score={best_score:.6f}")
                    break

        # ---------------- final / best ----------------
        if best_rho is None:
            with torch.no_grad():
                best_rho = rho.detach().clone()
                best_seeds = seeds.detach().clone()
                best_step = step
                best_score = float(L_total.detach().item())

        with torch.no_grad():
            final_shape_density = best_rho.clone()
            seed_points_final = self.generator.seeds_uv_to_xyz_nearest(best_seeds, uv, points_xyz)

            if mid_shape_density is None:
                mid_shape_density = final_shape_density.clone()
                seed_points_mid = seed_points_final

        print(f"FINAL RETURNED: best_step={best_step}, best_score={best_score:.6f}")

        return {
            "decoder": decoder,
            "ppnet": ppnet,
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
            "uv_init": uv_init,
        }

    def visualize_result_stepwise(self, result, points_xyz, faces_ijk):
        density_init_viz = self.viz.viz_normalize(result["Initial_shape_density"])
        density_mid_viz = self.viz.viz_normalize(result["Mid_shape_density"])
        density_fin_viz = self.viz.viz_normalize(result["Final_shape_density"])

        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        self.viz.plot_density_and_seedpoints_3stage(
            mesh_points=points_xyz.detach().cpu().numpy(),
            pv_faces=pv_faces_fixed,
            density_init=density_init_viz.detach().cpu().numpy(),
            density_mid=density_mid_viz.detach().cpu().numpy(),
            density_final=density_fin_viz.detach().cpu().numpy(),
            seed_points_init=result["seed_points_init"],
            seed_points_mid=result["seed_points_mid"],
            seed_points_final=result["seed_points_final"],
            shared_clim=False,
        )
    def visualize_result_final(self,  result,points_xyz, faces_ijk, thr=0.5,show_solid=True):
        density_fin_viz = self.viz.viz_normalize(result["Final_shape_density"])

        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        solid, thr_used,_ = self.viz.visualize_density_thresholded(
            points=points_xyz,
            pv_faces=pv_faces_fixed,
            density_total=density_fin_viz,
            thr=thr,          
            show_solid=show_solid
        )
