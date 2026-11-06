import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from learning.algorithms.behavior_cloning import BehaviorCloning, Hyperparameter
from opf_teacher import PairConfig, collect_pairs
from opf_teacher.opf.single_phase import load_case
from tools.dataset import configure_pv, select_period
from tools.features import FEATURES, TARGETS, encode

CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"


def main(config, output="results/train", seed=0):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    data = load_case(config["case"])
    case, period = select_period(data, config, default_steps=None)
    if "reward_eval" in config:
        reward_period = select_period(data, config | config["reward_eval"], default_steps=None)[1]
        if max(period["start_step"], reward_period["start_step"]) < min(
                period["start_step"] + period["steps"], reward_period["start_step"] + reward_period["steps"]):
            raise ValueError("Reward validation dates must not overlap the teacher training period")
    del data
    config |= period | {"voltage_source": "pre_action", "pad_initial": True}
    pairs = PairConfig(formulation="single_phase_socp", mode="local", device_id=config["device"],
                       window=config["window"], block_weeks=config.get("block_weeks"), pad_initial=config["pad_initial"])
    config["teacher_mode"] = "single_optimization" if pairs.block_weeks is None else "weekly_blocks"
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
    from tools.evaluation import PolicyEvaluator

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
        "reward_eval": {"start_date": "2023-03-01", "end_date": "2023-03-10"},
    }
    if config["architecture"] != "mlp" or config["training"] != "supervised":
        raise ValueError("Currently supported: architecture='mlp', training='supervised'")
    hyperparameters = Hyperparameter(hidden_size=64, learning_rate=1e-3, batch_size=64, epochs=100)
    output = "results/reward_selection"
    seed = 43
    max_iterations = 1

    output = Path(output) / f"seed_{seed}"
    x, y = main(config, output=output, seed=seed)
    train_split, eval_split = random_split(range(len(x)), [1 - config["eval_fraction"], config["eval_fraction"]],
                                         generator=torch.Generator().manual_seed(seed))
    x_train, y_train = x[train_split.indices], y[train_split.indices]
    x_eval, y_eval = x[eval_split.indices], y[eval_split.indices]
    evaluator = PolicyEvaluator(config | config["reward_eval"], seed=seed)
    trainer = BehaviorCloning(hyperparameters, reward_evaluator=lambda model: evaluator.evaluate(model)[1])
    for iteration in range(max_iterations):
        model = trainer.train(x_train, y_train, x_eval, y_eval, config,
                              output=output / f"trial_{iteration}", seed=seed)
    evaluator.env.close()
