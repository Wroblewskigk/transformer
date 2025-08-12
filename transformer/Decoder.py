from torch import nn
from transformer.LayerNormalization import LayerNormalization


class Decoder(nn.Module):
    """
    Decoder stacks N DecoderBlock layers and applies final normalization
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Sequentially applies each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Final normalization
        return self.norm(x)