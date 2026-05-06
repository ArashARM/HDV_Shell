import torch
import torch.nn as nn

from .anisotropy_predictor import AnisotropyPredictor
from .boundary_predictor import BoundaryPredictor
from .context_encoder import ContextEncoder
from .height_predictor import HeightPredictor
from .seed_identity import SeedIdentityEmbedding
from .seed_refiner import SeedRefiner
from .tau_predictor import TauPredictor
from .utils import check_finite
from .width_predictor import WidthPredictor


class PPNet(nn.Module):
    """
    PPNet predicts the full VoronoiDecoder control dictionary.

    The model uses one shared context backbone, then small specialized heads for
    seeds, pair widths, height, anisotropy, boundary parameters, and tau.
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

        self.context_encoder = ContextEncoder(context_dim, hidden)
        self.seed_identity = SeedIdentityEmbedding(self.n_seeds, self.seed_id_dim)
        self.seed_refiner = SeedRefiner(
            hidden=hidden,
            seed_id_dim=self.seed_id_dim,
            eps_uv=self.eps_uv,
            max_step_uv=self.max_step_uv,
            allow_seed_outside_domain=self.allow_seed_outside_domain,
            seed_domain_margin=self.seed_domain_margin,
            enable_checks=self.enable_checks,
        )
        self.width_predictor = WidthPredictor(
            hidden=hidden,
            freeze_w=self.freeze_w,
            w_const=self.w_const,
            w_head_bias_init=w_head_bias_init,
            enable_checks=self.enable_checks,
        )
        self.height_predictor = HeightPredictor(
            hidden=hidden,
            predict_height=self.predict_height,
            enable_checks=self.enable_checks,
        )
        self.anisotropy_predictor = AnisotropyPredictor(
            hidden=hidden,
            use_Metric_anisotropy=self.use_Metric_anisotropy,
            enable_checks=self.enable_checks,
        )
        self.boundary_predictor = BoundaryPredictor(
            hidden=hidden,
            predict_boundary_params=self.predict_boundary_params,
            enable_checks=self.enable_checks,
        )
        self.tau_predictor = TauPredictor(
            hidden=hidden,
            predict_tau=self.predict_tau,
            tau_pred_start=self.tau_pred_start,
            tau_pred_min=self.tau_pred_min,
            tau_pred_max=self.tau_pred_max,
            enable_checks=self.enable_checks,
        )

    # Compatibility properties for existing training code.
    @property
    def mlp(self):
        return self.context_encoder.mlp

    @property
    def seed_id_embed(self):
        return self.seed_identity.embedding

    @property
    def seed_refine(self):
        return self.seed_refiner.seed_refine

    @property
    def delta_head(self):
        return self.seed_refiner.delta_head

    @property
    def w_head(self):
        return self.width_predictor.w_head

    @property
    def h_head(self):
        return self.height_predictor.h_head

    @property
    def theta_head(self):
        return self.anisotropy_predictor.theta_head

    @property
    def a_head(self):
        return self.anisotropy_predictor.a_head

    @property
    def boundary_width_head(self):
        return self.boundary_predictor.boundary_width_head

    @property
    def boundary_alpha_head(self):
        return self.boundary_predictor.boundary_alpha_head

    @property
    def boundary_beta_head(self):
        return self.boundary_predictor.boundary_beta_head

    @property
    def tau_head(self):
        return self.tau_predictor.tau_head

    def _check(self, tensor, name):
        check_finite(tensor, name, self.enable_checks)

    def forward(self, context, uv_init, offset_scale=1.0):
        batch_size = context.shape[0]
        n_seeds = self.n_seeds
        eps_uv = self.eps_uv

        self._check(context, "context")

        # uv_init may be shared across the batch or already batched.
        if uv_init.dim() == 2:
            uv_init_b = uv_init.unsqueeze(0).expand(batch_size, -1, -1)
        elif uv_init.dim() == 3:
            uv_init_b = uv_init
        else:
            raise ValueError("uv_init must be (S,2) or (B,S,2)")

        if self.allow_seed_outside_domain:
            uv_base = uv_init_b
        else:
            uv_base = uv_init_b.clamp(eps_uv, 1.0 - eps_uv)
        self._check(uv_base, "uv_base")

        # context -> ContextEncoder -> z
        z = self.context_encoder(context)
        self._check(z, "z")

        # z + uv + seed_id_features -> SeedRefiner -> h, seeds_uv
        seed_id_features = self.seed_identity(batch_size, n_seeds, uv_base.device)
        self.seed_refiner.allow_seed_outside_domain = bool(self.allow_seed_outside_domain)
        self.seed_refiner.seed_domain_margin = float(self.seed_domain_margin)
        h, seeds_uv = self.seed_refiner(
            z,
            uv_base,
            seed_id_features=seed_id_features,
            offset_scale=offset_scale,
        )

        # h -> seed-level/pairwise predictors
        self.width_predictor.freeze_w = self.freeze_w
        self.width_predictor.w_const = self.w_const
        w_raw = self.width_predictor(h, batch_size, n_seeds, z)
        theta, a_raw = self.anisotropy_predictor(h)

        # z -> global predictors
        h_raw = self.height_predictor(z)
        boundary_width_raw, boundary_alpha_raw, boundary_beta_raw = self.boundary_predictor(z)
        tau = self.tau_predictor(z)

        return {
            "seeds_raw": seeds_uv,
            "w_raw": w_raw,
            "h_raw": h_raw,
            "theta": theta,
            "a_raw": a_raw,
            "boundary_width_raw": boundary_width_raw,
            "boundary_alpha_raw": boundary_alpha_raw,
            "boundary_beta_raw": boundary_beta_raw,
            "tau": tau,
        }
