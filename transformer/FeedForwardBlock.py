import torch
from torch import nn


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
