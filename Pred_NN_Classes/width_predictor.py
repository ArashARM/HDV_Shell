import torch
import torch.nn as nn

from .utils import check_finite


class WidthPredictor(nn.Module):
    """Predict symmetric pairwise widths, or return a fixed width matrix."""

    def __init__(
        self,
        hidden,
        freeze_w=False,
        w_const=0.25,
        w_head_bias_init=0.0,
        enable_checks=True,
    ):
        super().__init__()
        self.freeze_w = freeze_w
        self.w_const = w_const
        self.enable_checks = enable_checks

        self.w_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.w_head.weight)
        nn.init.constant_(self.w_head.bias, w_head_bias_init)

    def forward(self, h, batch_size, n_seeds, z):
        if self.freeze_w:
            w_raw = torch.full(
                (batch_size, n_seeds, n_seeds),
                self.w_const,
                device=z.device,
                dtype=z.dtype,
            )
        else:
            pair_h = 0.5 * (h.unsqueeze(2) + h.unsqueeze(1))
            w_raw = self.w_head(pair_h).squeeze(-1)
            w_raw = 0.5 * (w_raw + w_raw.transpose(1, 2))

        check_finite(w_raw, "w_raw", self.enable_checks)
        return w_raw
