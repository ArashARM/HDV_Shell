import torch
import torch.nn as nn


class SeedIdentityEmbedding(nn.Module):
    """Optional learned identity features for each seed index."""

    def __init__(self, n_seeds, seed_id_dim):
        super().__init__()
        self.seed_id_dim = int(seed_id_dim)
        if self.seed_id_dim > 0:
            self.embedding = nn.Embedding(n_seeds, self.seed_id_dim)
        else:
            self.embedding = None

    def forward(self, batch_size, n_seeds, device):
        if self.embedding is None:
            return None

        seed_ids = torch.arange(n_seeds, device=device, dtype=torch.long)
        return self.embedding(seed_ids).unsqueeze(0).expand(batch_size, -1, -1)
