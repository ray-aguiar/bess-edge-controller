import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from learning.networks.mlp import MLP


class Hyperparameter:
    def __init__(self, hidden_size=64, learning_rate=1e-3, batch_size=64, epochs=100):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs


class BehaviorCloning:
    """Train on teacher actions and select the lowest validation MSE."""

    def __init__(self, hyperparameters=None, network=MLP):
        self.hyperparameters = Hyperparameter() if hyperparameters is None else hyperparameters
        self.network = network

    def train(self, x_train, y_train, x_eval, y_eval, config, output, seed):
        params = self.hyperparameters
        if not len(x_train) or not len(x_eval) or params.epochs < 1:
            raise ValueError("Use nonempty training/evaluation sets and at least one epoch")
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        input_size = x_train[0].numel() if self.network is MLP else x_train.shape[-1]
        output_size = y_train.shape[-1]
        self.model = self.network(input_size=input_size, hidden_size=params.hidden_size,
                                  output_size=output_size).cpu()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate)
        loss_function = nn.MSELoss()
        generator = torch.Generator().manual_seed(seed)
        batches = DataLoader(TensorDataset(x_train, y_train), batch_size=params.batch_size,
                             shuffle=True, generator=generator)
        self.history = []
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        best_mse, best_epoch = float("inf"), None
        epochs = tqdm(range(params.epochs), desc="Epochs", position=0)
        for epoch in epochs:
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y in tqdm(batches, desc="Batches", position=1, leave=False):
                optimizer.zero_grad()
                loss = loss_function(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            metrics = {"epoch": epoch + 1, "train_mse": total_loss / len(x_train), "eval_mse": self.evaluate(x_eval, y_eval)}
            self.history.append(metrics)
            epochs.set_postfix(metrics)
            if metrics["eval_mse"] < best_mse:
                best_mse, best_epoch = metrics["eval_mse"], epoch + 1
                torch.save(self.model.state_dict(), output / "best.pt")

        torch.save(self.model.state_dict(), output / "weights.pt")
        report = {"network": self.network.__name__, "input_size": input_size, "output_size": output_size,
                  "hyperparameters": vars(params), "seed": seed, "config": config,
                  "best_epoch": best_epoch, "history": self.history}
        (output / "training.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        self.model.load_state_dict(torch.load(output / "best.pt", weights_only=True))
        return self.model

    def evaluate(self, x, y):
        """Return MSE on a separately prepared validation or test dataset."""
        self.model.eval()
        with torch.inference_mode():
            return nn.functional.mse_loss(self.model(x), y).item()
