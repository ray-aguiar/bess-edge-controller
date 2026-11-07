from torch import nn


class LSTM(nn.Module):
    """One recurrent layer followed by the current action prediction."""

    def __init__(self, input_size, hidden_size=64, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1])
