import json
import random
from pathlib import Path

import numpy as np
import torch

from learning.networks.mlp import MLP
from learning.networks.lstm import LSTM
from tools.evaluation import PolicyEvaluator
from tools.export_results import save_timeseries
from tools.generate_dataset import dataset_directory
from tools.plot_results import create_plots
from train import DATASET_CONFIG


CASE_PATH = Path(__file__).resolve().parent / "data" / "switzerland" / "5bus" #"5bus-switzerland"
ARCHITECTURE = "mlp"
TRAINING = "supervised"
MODEL_EXTRA_FEATURES = {} # Choose the same feature selection used in train.py.
# Examples:
# MODEL_EXTRA_FEATURES = {"time": ["hour_sin", "hour_cos", "month_sin", "month_cos"]}
# MODEL_EXTRA_FEATURES = {"grid": ["buy_price_01"]}
# MODEL_EXTRA_FEATURES = {"time": ["hour_sin", "hour_cos", "month_sin", "month_cos"],
#                         "grid": ["buy_price_01"]}
DAY = "2024-01-03" # Day data used for daily plots
SEED = 42
START_DATE = "2024-01-01"
END_DATE = "2024-01-10"


def main(model_path, config, output="results/test", seed=0):
    """Load a trained policy and run one environment episode on [start, end)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model_path = Path(model_path)
    training = json.loads(model_path.with_name("training.json").read_text(encoding="utf-8"))
    if training["config"].get("voltage_source") != "pre_action":
        raise ValueError("Expected a policy trained with pre-action voltage")
    network = {"mlp": MLP, "lstm": LSTM}[config["architecture"]]
    model = network(training["input_size"], training["hyperparameters"]["hidden_size"], training["output_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    config = config | {key: training["config"][key] for key in ("device", "window")} | {
        "extra_features": training["config"].get("extra_features", {}),
        "features": training["config"].get("features"),
        "label": training["config"].get("label"),
    }
    evaluator = PolicyEvaluator(config, seed=seed)
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


def test_output_directory(model_folder):
    relative_path = Path(model_folder).relative_to("results/train")
    return Path("results/test") / relative_path


if __name__ == "__main__":
    model_config = DATASET_CONFIG | {"extra_features": MODEL_EXTRA_FEATURES}
    model_folder = (Path("results/train") / TRAINING / ARCHITECTURE
                    / dataset_directory(model_config).name / f"seed_{SEED}" / "trial_0")
    model_path = model_folder / "best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model found for MODEL_EXTRA_FEATURES={MODEL_EXTRA_FEATURES} at {model_path}. "
            "Set MODEL_EXTRA_FEATURES to the training feature selection or train that configuration first."
        )
    config = {"case": str(CASE_PATH), "architecture": ARCHITECTURE, "training": TRAINING,
              "start_date": START_DATE, "end_date": END_DATE}
    output_dir = test_output_directory(model_folder)

    records, summary = main(model_path, config, output=output_dir, seed=SEED)
    create_plots(input_dir=output_dir, day=DAY)