import json
import random
from pathlib import Path

import numpy as np
import torch

from learning.networks.mlp import MLP
from tools.evaluation import PolicyEvaluator
from tools.export_results import save_timeseries


CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"


def main(model_path, config, output="results/test", seed=0):
    """Load a trained MLP and run one environment episode on [start, end)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model_path = Path(model_path)
    training = json.loads(model_path.with_name("training.json").read_text(encoding="utf-8"))
    if training["network"] != "MLP" or training["config"].get("voltage_source") != "pre_action":
        raise ValueError("Expected an MLP trained with pre-action voltage")
    model = MLP(training["input_size"], training["hyperparameters"]["hidden_size"], training["output_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    evaluator = PolicyEvaluator(config | {key: training["config"][key] for key in ("device", "window")}, seed=seed)
    records, summary = evaluator.evaluate(model, record=True)
    evaluator.env.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_timeseries(records, output)
    report = {"config": evaluator.config, "model": str(model_path), "seed": seed, "pv_policy": "fixed",
              "window": evaluator.config["window"], "device": evaluator.device, "voltage_source": "pre_action", "summary": summary}
    for name, data in (("records", records), ("summary", report)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return records, summary


if __name__ == "__main__":
    model_folder = Path("results/reward_selection/seed_43/trial_0")
    config = {"case": str(CASE_PATH), "start_date": "2024-01-01", "end_date": "2024-01-10"}
    seed = 42
    for name in ("best_mse", "best_reward"):
        records, summary = main(model_folder / f"{name}.pt", config,
                                output=Path("results/test") / model_folder.parent.name / model_folder.name / name, seed=seed)
