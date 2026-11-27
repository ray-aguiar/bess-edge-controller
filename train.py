"""Generate or reuse teacher data, then train a policy."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from tools.generate_dataset import dataset_directory, generate_dataset
from learning.algorithms.behavior_cloning import BehaviorCloning, Hyperparameter
from learning.networks.lstm import LSTM
from learning.networks.mlp import MLP

ROOT = Path(__file__).resolve().parent
DATASET_CONFIG = {
    "case": str(ROOT / "data" / "switzerland" / "5bus"),
    "device": "b1",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "window": 96,
    "block_weeks": 1,
    "extra_features": {}#{"time": ["hour_sin", "hour_cos", "month_sin", "month_cos"], "grid": ["buy_price_01"]},  # Use {} for no extra features.
}

ARCHITECTURE = "mlp"  # "mlp" or "lstm"
TRAINING = "supervised"
EVAL_FRACTION = 0.2
HYPERPARAMETERS = Hyperparameter(hidden_size=64, learning_rate=1e-3, batch_size=64, epochs=100)
SEED = 42
MAX_ITERATIONS = 1

def _dataset_path():
    path = dataset_directory(DATASET_CONFIG)
    if not all((path / name).is_file() for name in ("x.pt", "y.pt", "config.json")):
        print(f"Dataset not found at {path}; generating it now.")
        generate_dataset(DATASET_CONFIG, path)
    else:
        print(f"Using existing dataset: {path}")
    return path

def load_dataset(dataset_dir):
    """Load tensors and input/target metadata produced by tools/generate_dataset.py."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    metadata = json.loads((dataset_dir / "config.json").read_text(encoding="utf-8"))
    features = metadata["features"]
    targets = metadata["targets"]
    if metadata["config"].get("features") != features:
        raise ValueError("Dataset feature metadata is inconsistent")

    x = torch.load(dataset_dir / "x.pt", map_location="cpu", weights_only=True)
    y = torch.load(dataset_dir / "y.pt", map_location="cpu", weights_only=True)
    if x.ndim != 3 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected X [samples, window, features] and Y [samples, targets]")
    if x.shape[-1] != len(features) or y.shape[-1] != len(targets):
        raise ValueError("Dataset tensor dimensions do not match its feature/target metadata")
    return x, y, metadata

def main(dataset_dir=None, architecture=ARCHITECTURE, eval_fraction=EVAL_FRACTION,
         hyperparameters=None, seed=SEED, max_iterations=MAX_ITERATIONS,
         output_root=ROOT / "results" / "train"):
    if architecture not in {"mlp", "lstm"}:
        raise ValueError("architecture must be 'mlp' or 'lstm'")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    dataset_dir = _dataset_path() if dataset_dir is None else Path(dataset_dir)
    x, y, metadata = load_dataset(dataset_dir)
    config = metadata["config"] | {"architecture": architecture, "training": TRAINING,
                                   "eval_fraction": eval_fraction, "dataset_path": str(dataset_dir)}

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train_split, eval_split = random_split(range(len(x)), [1 - eval_fraction, eval_fraction],
                                          generator=torch.Generator().manual_seed(seed))
    x_train, y_train = x[train_split.indices], y[train_split.indices]
    x_eval, y_eval = x[eval_split.indices], y[eval_split.indices]

    trainer = BehaviorCloning(hyperparameters or HYPERPARAMETERS, network={"mlp": MLP, "lstm": LSTM}[architecture])
    output_dir = Path(output_root) / TRAINING / architecture / dataset_dir.name / f"seed_{seed}"

    return [trainer.train(x_train, y_train, x_eval, y_eval, config, output_dir / f"trial_{i}", seed)
            for i in range(max_iterations)]

if __name__ == "__main__":
    main()