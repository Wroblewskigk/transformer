import math
from torch import nn


class InputEmbeddingsBlock(nn.Module):
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
