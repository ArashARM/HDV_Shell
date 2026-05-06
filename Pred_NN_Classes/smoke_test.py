import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pred_NN_Classes import PPNet


def main():
    model = PPNet(context_dim=8, n_seeds=5)
    context = torch.randn(2, 8)
    uv_init = torch.rand(5, 2)
    out = model(context, uv_init)

    expected_keys = {
        "seeds_raw",
        "w_raw",
        "h_raw",
        "theta",
        "a_raw",
        "boundary_width_raw",
        "boundary_alpha_raw",
        "boundary_beta_raw",
        "tau",
    }
    assert set(out.keys()) == expected_keys
    print("from HDVClassnNet import PPNet works")
    print("out keys:", sorted(out.keys()))


if __name__ == "__main__":
    main()
    