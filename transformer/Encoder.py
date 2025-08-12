from torch import nn
from transformer.LayerNormalization import LayerNormalization


class Encoder(nn.Module):
    """
    Encoder stacks N EncoderBlock layers and applies final layer normalization
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        # Pass input through each encoder block in sequence
        for layer in self.layers:
            x = layer(x, mask)
        # Final layer normalization
        return self.norm(x)