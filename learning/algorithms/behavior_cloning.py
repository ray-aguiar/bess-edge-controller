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
    """MSE training; an optional reward_evaluator(model) returns rollout metrics."""

    def __init__(self, hyperparameters=None, network=MLP, reward_evaluator=None):
        self.hyperparameters = Hyperparameter() if hyperparameters is None else hyperparameters
        self.network = network
        self.reward_evaluator = reward_evaluator

    def train(self, x_train, y_train, x_eval, y_eval, config, output, seed):
        params = self.hyperparameters
        if not len(x_train) or not len(x_eval) or params.epochs < 1:
            raise ValueError("Use nonempty training/evaluation sets and at least one epoch")
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        input_size, output_size = x_train[0].numel(), y_train.shape[-1]
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
        best_mse, best_reward = float("inf"), -float("inf")
        best_epochs = {"mse": None, "reward": None}
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
            evaluation = self.reward_evaluator(self.model) if self.reward_evaluator is not None else None
            if evaluation is not None:
                metrics |= {"eval_reward": evaluation["reward"], "eval_soc_final": evaluation["soc_final_frac"]}
                if evaluation["reward"] > best_reward:
                    best_reward, best_epochs["reward"] = evaluation["reward"], epoch + 1
                    torch.save(self.model.state_dict(), output / "best_reward.pt")
            epochs.set_postfix(metrics)
            if metrics["eval_mse"] < best_mse:
                best_mse, best_epochs["mse"] = metrics["eval_mse"], epoch + 1
                for name in ("best.pt", "best_mse.pt"):
                    torch.save(self.model.state_dict(), output / name)

        torch.save(self.model.state_dict(), output / "weights.pt")
        report = {"network": self.network.__name__, "input_size": input_size, "output_size": output_size,
                  "hyperparameters": vars(params), "seed": seed, "config": config,
                  "best_epochs": best_epochs, "history": self.history}
        (output / "training.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        self.model.load_state_dict(torch.load(output / "best.pt", weights_only=True))
        return self.model

    def evaluate(self, x, y):
        """Return MSE on a separately prepared validation or test dataset."""
        self.model.eval()
        with torch.inference_mode():
            return nn.functional.mse_loss(self.model(x), y).item()
