import torch
from torch import nn


class LayerNormalization(nn.Module):
    """
    Performs layer normalization for better training stability and convergence
    """

    def __init__(self, features: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        # Epsilon for numerical stability and to avoid division by zero
        self.epsilon = epsilon
        # Learnable gain parameter
        self.alpha = nn.Parameter(torch.ones(features))
        # Learnable bias parameter
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias