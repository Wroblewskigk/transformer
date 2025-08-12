from torch import nn
from transformer.FeedForwardBlock import FeedForwardBlock
from transformer.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from transformer.ResidualConnection import ResidualConnection


class DecoderBlock(nn.Module):
    """
    DecoderBlock adds masked self-attention, cross-attention, and a feed-forward block,
    each with its own residual connection
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Three residual connections, each after: self-attention, cross-attention, feedforward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    # noinspection PyShadowingNames
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked self-attention for tgt sequence
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # Cross-attention: queries from decoder, keys/values from encoder
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        # Position-wise feed-forward
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x