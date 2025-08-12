import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding adds information about token position for the model to distinguish token order
    """

    def __init__(self, dmodel: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        positional_encoding = torch.zeros(seq_len, dmodel)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel)
        )
        # Apply sines to even indices in the dimension, cosines to odd indices
        positional_encoding[:, 0::2] = torch.sin(position * denominator)
        positional_encoding[:, 1::2] = torch.cos(position * denominator)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        x = x + (self.positional_encoding[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)