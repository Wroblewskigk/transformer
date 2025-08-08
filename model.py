import math

import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, dmodel: int, vocabulary_size: int):
        super().__init__()
        self.dmodel = dmodel
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, dmodel)

        def forward(self, x):
            return self.embedding(x) * math.sqrt(self.dmodel)
