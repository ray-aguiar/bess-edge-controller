from collections import deque
import json
import random
from pathlib import Path

import numpy as np
import torch

from learning.networks.mlp import MLP
from opendss_environment import EnvironmentConfig, MicrogridEnv
from opf_teacher.opf.single_phase import load_case
from tools.dataset import select_period
from tools.export_results import save_timeseries
from tools.features import TARGETS, encode, local_observation


CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"


def main(model_path, config, output="results/test", seed=0):
    """Load a trained MLP and run one environment episode on [start, end)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model_path = Path(model_path)
    training = json.loads(model_path.with_name("training.json").read_text(encoding="utf-8"))
    if training["network"] != "MLP":
        raise ValueError("This test currently loads MLP models only")
    if training["config"].get("voltage_source") != "pre_action":
        raise ValueError("Regenerate the dataset and retrain using pre-action voltage before testing")
    model = MLP(training["input_size"], training["hyperparameters"]["hidden_size"], training["output_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    case, period = select_period(load_case(config["case"]), config, default_steps=None)
    config = config | period
    device = training["config"]["device"]
    if len(case.bess) != 1 or case.bess[0].id != device:
        raise ValueError("The case must contain one BESS matching the trained device")
    bus = case.buses[case.bess[0].bus].name
    history = deque(maxlen=training["config"]["window"])
    env = MicrogridEnv(EnvironmentConfig(case.config_path, episode_steps=config["steps"],
                                        start_step=config["start_step"]))
    observation, _ = env.reset(seed=seed)
    records = []
    for step in range(config["steps"]):
        state = local_observation(observation, bus, device)
        history.append(state)
        window = [history[0]] * (history.maxlen - len(history)) + list(history)
        with torch.inference_mode():
            predicted = model(encode(window).unsqueeze(0)).reshape(-1)
        if predicted.numel() != len(TARGETS):
            raise ValueError("The model must return active and reactive BESS power")
        action = {
            "bess": {device: dict(zip(TARGETS, predicted.tolist()))},
            "pv": {key: {"generation_pu": sum(values["available_pu"].values()), "q_injection_pu": 0.0}
                   for key, values in observation["pv"].items()},
        }
        observation, reward, terminated, truncated, info = env.step(action)
        records.append({"state": state, "action": action, "info": info})
        if terminated or truncated:
            break
    env.close()

    summary = {"steps": len(records), "cost_chf": sum(row["info"]["cost"] for row in records),
               "reward": sum(row["info"]["reward"] for row in records)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_timeseries(records, output)
    report = {"config": config, "model": str(model_path), "seed": seed, "pv_policy": "fixed",
              "window": history.maxlen, "device": device, "voltage_source": "pre_action", "summary": summary}
    (output / "records.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return records, summary


if __name__ == "__main__":
    model_path = "results/train/seed_42/trial_0/weights.pt"
    config = {"case": str(CASE_PATH), "start_date": "2024-01-01", "end_date": "2024-01-02"}
    seed = 42
    records, summary = main(model_path, config, output="results/test/seed_42/trial_0", seed=seed)
