from torch import nn
from transformer.FeedForwardBlock import FeedForwardBlock
from transformer.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from transformer.ResidualConnection import ResidualConnection


class EncoderBlock(nn.Module):
    """
    EncoderBlock combines self-attention and feed-forward blocks with residual connections
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections, each after: self-attention, feed-forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    # noinspection PyShadowingNames
    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x