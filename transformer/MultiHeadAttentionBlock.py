import math
from torch import nn


class MultiHeadAttentionBlock(nn.Module):
    """
    MultiHeadAttentionBlock performs self-/cross-attention in parallel over multiple heads
    """

    def __init__(self, dmodel: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.num_heads = num_heads
        assert dmodel % num_heads == 0, "dmodel must be divisible by num_heads"

        self.dk = dmodel // num_heads
        self.wq = nn.Linear(dmodel, dmodel, bias=False)
        self.wk = nn.Linear(dmodel, dmodel, bias=False)
        self.wv = nn.Linear(dmodel, dmodel, bias=False)
        self.wo = nn.Linear(dmodel, dmodel, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # Dimension per head
        dk = query.shape[-1]
        # Scaled dot-product attention
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            # Masks out positions, that shouldn't be considered in calculating attention
            attention_scores.masked_fill_(mask == 0, -math.inf)
        # Normalize to probabilities
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            # Optionally apply dropout to attention distribution
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        query = query.view(
            query.shape[0], query.shape[1], self.num_heads, self.dk
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.dk).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.num_heads, self.dk
        ).transpose(1, 2)

        # noinspection PyAttributeOutsideInit
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Concatenates all heads and projects the result
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.dk)
        )

        return self.wo(x)