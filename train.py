import json
from pathlib import Path

import torch
from opf_teacher.opf.single_phase import load_case

from tools.features import FEATURES, TARGETS, encode
from tools.dataset import select_period
from tools.teacher_data import collect_pairs


CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"
DEFAULT_CONFIG = {"window": 3, "device": "b1", "horizon": 96, "case": str(CASE_PATH),
                  "start_date": None, "end_date": None}


def main(config=None, output="results/train", seed=0, *, trainer=None):
    """Collect CPU X/Y tensors; return the trained model when a trainer is supplied."""
    config = DEFAULT_CONFIG | (config or {})
    torch.manual_seed(seed)
    case, period = select_period(load_case(config["case"]), config, config["horizon"])
    config |= period
    x, y = collect_pairs(case, config)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "seed": seed, "features": FEATURES, "targets": TARGETS,
                "state_source": "teacher", "voltage_source": "previous_interval", "pv_policy": "fixed"}
    for name, data in (("x", x), ("y", y), ("config", metadata)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    x = torch.stack([encode(window) for window in x])
    y = torch.tensor([[action[name] for name in TARGETS] for action in y], dtype=torch.float32)
    torch.save(x, output / "x.pt")
    torch.save(y, output / "y.pt")
    print("X:", tuple(x.shape), "Y:", tuple(y.shape))
    if trainer is not None:
        return trainer(x, y, config, output=output, seed=seed)
    return x, y


if __name__ == "__main__":
    config = {
        "architecture": "mlp",
        "training": "supervised",
        "window": 3,
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
    }
    output = "results/train"
    seed = 42
    trainer = None
    result = main(config, output=output, seed=seed, trainer=trainer)
