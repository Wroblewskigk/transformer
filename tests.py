"""
Embedding class inner workings tests
"""

import torch
from torch import nn

device = "cuda"
print(torch.device(device))

embedding = nn.Embedding(3, 5)

inputTensor = torch.LongTensor(
    [
        [[1, 2], [1, 2], [1, 2]],
        [[1, 2], [1, 2], [0, 0]],
        [[1, 2], [1, 2], [1, 2]],
        [[1, 2], [1, 2], [0, 0]],
    ]
)

print(embedding(inputTensor))
