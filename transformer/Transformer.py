import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    InputEmbeddings maps token indices to dense vectors of a fixed dimension (dmodel)
    """

    def __init__(self, dmodel: int, vocab_size: int) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dmodel)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dmodel)


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


class FeedForwardBlock(nn.Module):
    """
    FeedForwardBlock provides a position-wise two-layer fully connected network used in transformers
    """

    def __init__(self, dmodel: int, dff: int, dropout: float) -> None:
        super().__init__()
        # First linear layer increases dimensionality
        self.linear_1 = nn.Linear(dmodel, dff)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        # Second linear layer projects back to dmodel
        self.linear_2 = nn.Linear(dff, dmodel)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


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


class ProjectionLayer(nn.Module):
    """
    ProjectionLayer maps the transformer output dimension back to the vocabulary space
    """
    def __init__(self, dmodel, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(dmodel, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)


class Transformer(nn.Module):
    """
    The full Transformer model tying together all components
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    dmodel: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    dff: int = 2048,
) -> Transformer:
    """
    Factory function to build a full Transformer model with given hyperparameters
    :param src_vocab_size: Source language vocabulary size
    :param tgt_vocab_size: Target language vocabulary size
    :param src_seq_len: Source token max sequence length
    :param tgt_seq_len: Target token max sequence length
    :param dmodel: Transformer dmodel hyperparameter
    :param N: Transformer N hyperparameter
    :param h: Transformer number of heads hyperparameter
    :param dropout: Transformer dropout hyperparameter
    :param dff: Transformer DFF hyperparameter
    :return: Transformer model ready for training
    """

    # Create the embedding layers
    src_embed = InputEmbeddings(dmodel, src_vocab_size)
    tgt_embed = InputEmbeddings(dmodel, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(dmodel, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(dmodel, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(dmodel, h, dropout)
        feed_forward_block = FeedForwardBlock(dmodel, dff, dropout)
        encoder_block = EncoderBlock(
            dmodel, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(dmodel, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(dmodel, h, dropout)
        feed_forward_block = FeedForwardBlock(dmodel, dff, dropout)
        decoder_block = DecoderBlock(
            dmodel,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(dmodel, nn.ModuleList(encoder_blocks))
    decoder = Decoder(dmodel, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(dmodel, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
