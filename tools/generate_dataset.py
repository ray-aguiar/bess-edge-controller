"""Generate reusable teacher X/Y tensors for one dataset configuration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

from opf_teacher import PairConfig, collect_pairs
from opf_teacher.opf.single_phase import load_case

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dataset import configure_pv, select_period
from tools.features import TARGETS, encode, feature_order

ROOT = Path(__file__).resolve().parents[1]

def _path_token(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "all"

def dataset_directory(config):
    """Build a compact, stable output path for the full dataset configuration."""
    case_path = Path(config["case"])
    case_name = "_".join(case_path.parts[-2:]) if len(case_path.parts) > 1 else case_path.name
    feature_group = "extra_features" if config.get("extra_features") else "no_extra_features"
    year = _path_token(str(config.get("start_date") or "all")[:4])
    window = "flat" if config.get("window") is None else f"w{config['window']}"
    suffix = "extra" if config.get("extra_features") else "no_extra"
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    digest = hashlib.sha256(payload).hexdigest()[:8]
    name = f"{year}_{_path_token(config['device'])}_{window}_{suffix}_{digest}"
    return ROOT / "results" / "datasets" / case_name / name

def generate_dataset(config, output_dir=None):
    """Collect and save teacher pairs plus encoded CPU tensors."""
    config = deepcopy(config)
    config.setdefault("extra_features", {})
    output_dir = dataset_directory(config) if output_dir is None else Path(output_dir)
    case, period = select_period(load_case(config["case"]), config, default_steps=None)
    config.update(period, voltage_source="pre_action", pad_initial=True)
    config["label"] = "extra_features" if config["extra_features"] else None
    pairs = PairConfig(formulation="single_phase_socp", mode="local", device_id=config["device"],
                       window=config["window"], block_weeks=config.get("block_weeks"),
                       pad_initial=True, extra_features=config["extra_features"])
    config["teacher_mode"] = "single_optimization" if config["block_weeks"] is None else "weekly_blocks"
    x_pairs, y_pairs = collect_pairs(configure_pv(case, "fixed"), pairs)
    config["features"] = feature_order(x_pairs[0][-1])

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "features": config["features"], "targets": TARGETS,
                "state_source": "teacher", "voltage_source": config["voltage_source"], "pv_policy": "fixed"}
    for name, data in (("x", x_pairs), ("y", y_pairs), ("config", metadata)):
        (output_dir / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    x = torch.stack([encode(window, config["features"]) for window in x_pairs])
    y = torch.tensor([[action[name] for name in TARGETS] for action in y_pairs], dtype=torch.float32)
    torch.save(x, output_dir / "x.pt")
    torch.save(y, output_dir / "y.pt")
    print(f"Generated X{tuple(x.shape)}, Y{tuple(y.shape)} at {output_dir}")
    return output_dir