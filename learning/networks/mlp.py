from torch import nn


class MLP(nn.Module):
    """One hidden layer for batched observation windows."""

    def __init__(self, input_size, hidden_size=64, output_size=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)
