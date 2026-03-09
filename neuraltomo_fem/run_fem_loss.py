import torch

from neuraltomo_fem.FE import FE

def fiber_dir_to_angles(fiber_dir: torch.Tensor):
    """
    fiber_dir: (nele,3)
    returns phi, theta each (nele,)
    Matches NeuralTOMO angle convention.
    """
    f = fiber_dir / (torch.linalg.norm(fiber_dir, dim=1, keepdim=True) + 1e-12)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    fxz = torch.sqrt(fx * fx + fz * fz) + 1e-12
    theta = torch.atan2(fy, fxz)
    phi = torch.atan2(fz, fx)
    return phi, theta

class NeuralTOMOFEM:
    """
    Minimal callable FEM:
      stress, compliance = fem(density, fiber_dir)
    """
    def __init__(self, problem, device="cpu", isotropic=False):
        """
        `problem` is whatever FE(...) expects in their code.
        In NeuralTOMO this comes from their settings factory.
        If you don’t want their settings system, we can build a tiny Problem class later.
        """
        self.device = torch.device(device)
        self.fe = FE(problem, device=str(self.device))
        self.isotropic = isotropic

    def __call__(self, density: torch.Tensor, fiber_dir: torch.Tensor, penal=3):
        density = density.to(self.device).flatten().float()
        fiber_dir = fiber_dir.to(self.device).reshape(-1, 3).float()

        phi, theta = fiber_dir_to_angles(fiber_dir)

        # exact call used in NeuralTOMO:
        stress, compliance = self.fe.solve_stress_new(
            phi, theta, density, penal=penal, isotropic=self.isotropic
        )
        return stress, compliance