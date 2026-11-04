
from tools.features import FEATURES, TARGETS, encode
from opf_teacher import PairConfig, collect_pairs
from opf_teacher.opf.single_phase import load_case
from tools.dataset import configure_pv, select_period

from learning.algorithms.behavior_cloning import BehaviorCloning, Hyperparameter


from pathlib import Path
import random
import numpy as np
import torch
import json


CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"


def main(config, output="results/train", seed=0):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    case, period = select_period(load_case(config["case"]), config, config["horizon"])
    config |= period | {"voltage_source": "pre_action"}
    pairs = PairConfig(formulation="single_phase_socp", mode="local", device_id=config["device"],
                       window=config["window"], horizon=config["horizon"])
    x, y = collect_pairs(configure_pv(case, "fixed"), pairs)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "seed": seed, "features": FEATURES, "targets": TARGETS,
                "state_source": "teacher", "voltage_source": config["voltage_source"], "pv_policy": "fixed"}
    for name, data in (("x", x), ("y", y), ("config", metadata)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    x = torch.stack([encode(window) for window in x])
    y = torch.tensor([[action[name] for name in TARGETS] for action in y], dtype=torch.float32)
    torch.save(x, output / "x.pt")
    torch.save(y, output / "y.pt")
    print("X:", tuple(x.shape), "Y:", tuple(y.shape))
    return x, y


if __name__ == "__main__":
    config = {
        "case": str(CASE_PATH),
        "device": "b1",
        "horizon": 96,
        "start_date": "2023-01-01",
        "end_date": "2023-01-02",
        "architecture": "mlp",
        "training": "supervised",
        "window": 3,
    }
    hyperparameters = Hyperparameter(hidden_size=64, learning_rate=1e-3, batch_size=64, epochs=100)
    output = "results/train"
    seed = 42
    max_iterations = 1

    output = Path(output) / f"seed_{seed}"
    x, y = main(config, output=output, seed=seed)
    trainer = BehaviorCloning(hyperparameters)
    for iteration in range(max_iterations):
        model = trainer.train(x, y, config, output=output / f"trial_{iteration}", seed=seed)
