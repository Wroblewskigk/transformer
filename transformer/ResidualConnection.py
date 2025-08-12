from torch import nn
from transformer.LayerNormalization import LayerNormalization


class ResidualConnection(nn.Module):
    """
    Implements residual connections followed by layer normalization
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))