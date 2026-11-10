import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from learning.algorithms.behavior_cloning import BehaviorCloning, Hyperparameter
from learning.networks.lstm import LSTM
from learning.networks.mlp import MLP
from opf_teacher import PairConfig, collect_pairs
from opf_teacher.opf.single_phase import load_case
from tools.dataset import configure_pv, select_period
from tools.features import TARGETS, encode, feature_order

CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"


def main(config, output="results/train", seed=0):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    case, period = select_period(load_case(config["case"]), config, default_steps=None)
    config |= period | {"voltage_source": "pre_action", "pad_initial": True,
                        "label": "extra_features" if config.get("extra_features") else None}
    pairs = PairConfig(formulation="single_phase_socp", mode="local", device_id=config["device"],
                       window=config["window"], block_weeks=config.get("block_weeks"), pad_initial=config["pad_initial"],
                       extra_features=config.setdefault("extra_features", {}))
    config["teacher_mode"] = "single_optimization" if pairs.block_weeks is None else "weekly_blocks"
    x, y = collect_pairs(configure_pv(case, "fixed"), pairs)
    config["features"] = feature_order(x[0][-1])

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "seed": seed, "features": config["features"], "targets": TARGETS,
                "state_source": "teacher", "voltage_source": config["voltage_source"], "pv_policy": "fixed"}
    for name, data in (("x", x), ("y", y), ("config", metadata)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    x = torch.stack([encode(window, config["features"]) for window in x])
    y = torch.tensor([[action[name] for name in TARGETS] for action in y], dtype=torch.float32)
    torch.save(x, output / "x.pt")
    torch.save(y, output / "y.pt")
    print("X:", tuple(x.shape), "Y:", tuple(y.shape))
    return x, y


if __name__ == "__main__":
    # mlp, lstm
    config = {
        "case": str(CASE_PATH),
        "device": "b1",
        "start_date": "2023-01-01",
        "end_date": "2023-02-28",
        "architecture": "mlp",
        "training": "supervised",
        "window": 96,
        "block_weeks": None,
        "eval_fraction": 0.2,
        "extra_features": {},#{"branches": ["p_flow_pu"]},
    }
    hyperparameters = Hyperparameter(hidden_size=64, learning_rate=1e-3, batch_size=64, epochs=100)
    output = f"results/train/{config['training']}/{config['architecture']}/"
    if config["extra_features"]:
        output = Path(output) / "extra_features"
    seed = 42
    max_iterations = 1

    output = Path(output) / f"seed_{seed}"
    x, y = main(config, output=output, seed=seed)
    train_split, eval_split = random_split(range(len(x)), [1 - config["eval_fraction"], config["eval_fraction"]],
                                         generator=torch.Generator().manual_seed(seed))
    x_train, y_train = x[train_split.indices], y[train_split.indices]
    x_eval, y_eval = x[eval_split.indices], y[eval_split.indices]
    network = {"mlp": MLP, "lstm": LSTM}[config["architecture"]]
    trainer = BehaviorCloning(hyperparameters, network=network)
    for iteration in range(max_iterations):
        model = trainer.train(x_train, y_train, x_eval, y_eval, config,
                              output=output / f"trial_{iteration}", seed=seed)
