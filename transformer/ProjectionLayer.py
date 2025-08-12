from torch import nn


class ProjectionLayer(nn.Module):
    """
    ProjectionLayer maps the transformer output dimension back to the vocabulary space
    """
    def __init__(self, dmodel, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(dmodel, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)