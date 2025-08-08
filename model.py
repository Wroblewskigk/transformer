import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, dmodel: int, vocabulary_size: int) -> None:
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


class LayerNormalization(nn.Module):
    def __init__(self, dmodel: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1, dmodel))  # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1, dmodel))  # Additive

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


class FeedForward(nn.Module):
    def __init__(self, dmodel: int, dff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dmodel, dff)  # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, dmodel)  # W2 and b2

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_scores = None  #########################################################################DIFFERENT
        self.dmodel = dmodel
        self.num_heads = num_heads
        assert num_heads % dmodel == 0, "num_heads must be divisible by dmodel"
        self.num_heads = (
            num_heads / dmodel
        )  ####################################################################################################DIFFERENT
        self.wq = nn.Linear(dmodel, dmodel)
        self.wk = nn.Linear(dmodel, dmodel)
        self.wv = nn.Linear(dmodel, dmodel)
        self.wo = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        dk = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -math.inf)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        """
        Q, K, V transpose shape goes like this: 
        
        [batch, sequence_length, dmodel] -> 
        [batch, sequence_length, num_heads, dk] -> 
        [batch, num_heads, dmodel]
        """
        query = query.view(
            query.shape[0],
            query.shape[1],
            self.num_heads,
            self.dk,
        ).transpose(1, 2)
        key = key.view(
            key.shape[0],
            key.shape[1],
            self.num_heads,
            self.dk,
        ).transpose(1, 2)
        value = value.view(
            value.shape[0],
            value.shape[1],
            self.num_heads,
            self.dk,
        ).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.dk)
        )
        return self.wo(x)


class ResidualConnection(nn.Module):
    def __init__(self, dmodel, dropout: float) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(
            dmodel
        )  ####################################################################################################DIFFERENT

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttention,
        feedforward: FeedForward,
        dropout: float,
        dmodel,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feedforward = feedforward
        self.residual_connection = nn.ModuleList(
            [
                ResidualConnection(dmodel, dropout) for _ in range(2)
            ]  ################################################################################################DIFFERENT
        )

    # noinspection PyShadowingNames
    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feedforward)
        return x


class Encoder(nn.Module):
    def __init__(self, dmodel, layers: nn.ModuleList) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.layers = layers
        self.norm = LayerNormalization(dmodel)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feedforward: FeedForward,
        dropout: float,
        dmodel,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dmodel, dropout) for _ in range(3)]
        )

    # noinspection PyShadowingNames
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention(x, x, x, target_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask),
        )
        x = self.residual_connection[2](x, self.feedforward)
        return x


class Decoder(nn.Module):
    def __init__(self, dmodel, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(
            dmodel
        )  ####################################################################################################DIFFERENT

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
