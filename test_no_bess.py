import json
import random
from pathlib import Path

import numpy as np
import torch

from tools.evaluation import PolicyEvaluator
from tools.export_results import save_timeseries


CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"

def main(config, model_folder, output="results/test/no_bess", seed=0):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    training = json.loads((Path(model_folder) / "training.json").read_text(encoding="utf-8"))

    config = config | {key: training["config"][key] for key in ("device", "window")} | {
        "extra_features": training["config"].get("extra_features", {}),
        "features": training["config"].get("features"),
        "label": training["config"].get("label"),
    }

    evaluator = PolicyEvaluator(config, seed=seed)
    records, summary = evaluator.evaluate(model=None, record=True, disable_bess=True)
    evaluator.env.close()

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_timeseries(records, output)

    report = {"config": evaluator.config, "seed": seed, "pv_policy": "fixed",
              "bess_policy": "disabled", "summary": summary}

    (output / "records.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return records, summary


if __name__ == "__main__":
    config = {
        "case": str(CASE_PATH),
        "architecture": "mlp",
        "training": "supervised",
        "start_date": "2024-01-01",
        "end_date": "2024-01-10",
    }

    model_folder = Path("results/train/supervised/mlp/seed_42/trial_0")

    records, summary = main(config, model_folder=model_folder, output="results/test/other/no_bess/seed_42/trial_0", seed=42)