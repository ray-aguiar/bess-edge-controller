import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learning.networks.mlp import MLP


class Hyperparameter:
    def __init__(self, hidden_size=64, learning_rate=1e-3, batch_size=64, epochs=100):
        self.learning_rate  = learning_rate
        self.hidden_size    = hidden_size
        self.batch_size     = batch_size
        self.epochs         = epochs


class BehaviorCloning:
    """Supervised imitation with manually supplied hyperparameters."""

    def __init__(self, hyperparameters=None, network=MLP):
        self.hyperparameters = Hyperparameter() if hyperparameters is None else hyperparameters
        self.network = network

    def train(self, x, y, config, output, seed):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        params = self.hyperparameters
        if params.epochs < 1:
            raise ValueError("epochs must be positive")
        input_size, output_size = x[0].numel(), y.shape[-1]
        self.model = self.network(input_size=input_size, hidden_size=params.hidden_size,
                                  output_size=output_size).cpu()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate)
        loss_function = nn.MSELoss()
        generator = torch.Generator().manual_seed(seed)
        batches = DataLoader(TensorDataset(x, y), batch_size=params.batch_size,
                             shuffle=True, generator=generator)
        self.history = []
        self.model.train()
        for epoch in range(params.epochs):
            total_loss = 0.0
            for batch_x, batch_y in batches:
                optimizer.zero_grad()
                loss = loss_function(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            self.history.append({"epoch": epoch + 1, "train_mse": total_loss / len(x)})

        self.model.eval()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output / "weights.pt")
        report = {"network": self.network.__name__, "input_size": input_size, "output_size": output_size,
                  "hyperparameters": vars(params), "seed": seed, "config": config, "history": self.history}
        (output / "training.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return self.model

    def evaluate(self, x, y):
        """Return MSE on a separately prepared validation or test dataset."""
        self.model.eval()
        with torch.inference_mode():
            return nn.functional.mse_loss(self.model(x), y).item()
