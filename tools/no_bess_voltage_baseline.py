"""Generate and cache a no-BESS-dispatch voltage baseline for a case."""
import json
from pathlib import Path

import pandas as pd

from opendss_environment import EnvironmentConfig, MicrogridEnv
from opf_teacher.opf.single_phase import load_case
from tools.dataset import select_period


ROOT = Path(__file__).resolve().parents[1]
START_DATE = "2023-01-01"
END_DATE = "2023-03-01"


def voltage_dataset_directory(case_path):
    """Return the shared voltage-baseline directory for a case."""
    parts = Path(case_path).resolve().parts
    case_name = "_".join(parts[-2:]) if len(parts) > 1 else parts[-1]
    return ROOT / "results" / "datasets" / case_name / "no_bess_voltages"


def generate_no_bess_voltages(
    case_path,
    start_date=START_DATE,
    end_date=END_DATE,
    output_dir=None,
):
    """Run fixed PV with zero BESS dispatch and save every bus/phase voltage."""
    case, period = select_period(
        load_case(case_path),
        {"start_date": start_date, "end_date": end_date},
        default_steps=None,
    )
    output_dir = voltage_dataset_directory(case_path) if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = MicrogridEnv(EnvironmentConfig(
        case.config_path,
        episode_steps=period["steps"],
        start_step=period["start_step"],
    ))
    rows = []
    try:
        observation, _ = env.reset()
        for _ in range(period["steps"]):
            row = {"timestamp": observation["timestamp"]}
            for bus, values in observation["buses"].items():
                for phase, voltage in values["v_before_pu"].items():
                    row[f"{bus}_{phase}_pu"] = voltage
            rows.append(row)

            action = {
                "bess": {
                    device: {"p_net_pu": 0.0, "q_injection_pu": 0.0}
                    for device in observation["bess"]
                },
                "pv": {
                    device: {
                        "generation_pu": sum(values["available_pu"].values()),
                        "q_injection_pu": 0.0,
                    }
                    for device, values in observation["pv"].items()
                },
            }
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    if len(rows) != period["steps"]:
        raise RuntimeError(
            f"Baseline produced {len(rows)} voltage rows; expected {period['steps']}"
        )

    voltage_path = output_dir / "voltages.csv"
    pd.DataFrame(rows).to_csv(voltage_path, index=False)
    metadata = {
        "case": str(Path(case_path).resolve()),
        "start_date": period["start_date"],
        "end_date": period["end_date"],
        "steps": len(rows),
        "pv_policy": "fixed",
        "bess_policy": "zero_dispatch",
        "voltage_units": "pu",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return voltage_path
