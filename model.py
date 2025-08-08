import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, dmodel: int, vocabulary_size: int):
        super().__init__()
        self.dmodel = dmodel
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, dmodel)

        def forward(self, x):
            return self.embedding(x) * math.sqrt(self.dmodel)


class PositionalEncoding(nn.Module):
    def __init__(self, dmodel: int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        """
        Usually you would use equations from paper "Attention is all you need", 
        however they have been optimized lately (or so i've heard)
        
        Original positional encoding for even indices (2i).
        PE(pos, 2i) = sin(pos / (10000**(2i / d_model)))
        Original positional encoding for odd indices (2i + 1).
        PE(pos, 2i+1) = cos(pos / (10000**(2i / d_model)))
        """

        # Matrix of shape (sequence_length, dmodel)
        positional_encoding = torch.zeros(sequence_length, dmodel)
        # Matrix of shape (sequence_length, 1)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel)
        )
        # Apply sin and cos to positional encoding
        positional_encoding[:, 0::2] = torch.sin(position * denominator)
        positional_encoding[:, 1::2] = torch.cos(position * denominator)

        # Make positional encoding (1, sequence_length, dmodel)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor):
        x = x + (self.positional_encoding[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
