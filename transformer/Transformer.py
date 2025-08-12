import torch
import torch.nn as nn

from transformer.Decoder import Decoder
from transformer.DecoderBlock import DecoderBlock
from transformer.Encoder import Encoder
from transformer.EncoderBlock import EncoderBlock
from transformer.FeedForwardBlock import FeedForwardBlock
from transformer.InputEmbeddingBlock import InputEmbeddingsBlock
from transformer.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from transformer.PositionalEncoding import PositionalEncoding
from transformer.ProjectionLayer import ProjectionLayer


class Transformer(nn.Module):
    """
    The full Transformer model tying together all components
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddingsBlock,
        tgt_embed: InputEmbeddingsBlock,
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
    src_embed = InputEmbeddingsBlock(dmodel, src_vocab_size)
    tgt_embed = InputEmbeddingsBlock(dmodel, tgt_vocab_size)

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
