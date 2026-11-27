"""Proportional BESS dispatch around the no-dispatch mean bus voltage."""
from pathlib import Path

import pandas as pd

from opf_teacher.opf.single_phase import load_case
from tools.no_bess_voltage_baseline import (generate_no_bess_voltages, voltage_dataset_directory)

K = 5000.0  # Requested active power in kW per pu voltage error.
BASELINE_PHASE = "a"

class VoltageProportionalStrategy:
    """Discharge below the baseline mean and charge above it."""

    def __init__(self, reference_voltage_pu, power_base_kva, gain_kw_per_pu=K):
        self.reference_voltage_pu = reference_voltage_pu
        self.power_base_kva = power_base_kva
        self.gain_kw_per_pu = gain_kw_per_pu

    def reset(self):
        pass

    def action(self, state, _p_charge_max_pu, _p_discharge_max_pu):
        voltage = float(state["bus"]["v_before_pu"])
        power_kw = self.gain_kw_per_pu * (
            voltage - self.reference_voltage_pu
        )
        return power_kw / self.power_base_kva, 0.0


def create_strategy(config):
    """Load or generate the 2023 baseline, then build the proportional rule."""
    case_path = config["case"]
    device_id = config["device"]
    case = load_case(case_path)
    bess = next((device for device in case.bess if device.id == device_id), None)
    if bess is None:
        raise ValueError(f"No BESS with ID {device_id!r} exists in {case_path!r}")
    bus = case.buses[bess.bus].name

    baseline_dir = voltage_dataset_directory(case_path)
    voltage_path = baseline_dir / "voltages.csv"
    if not voltage_path.exists():
        print(f"Voltage dataset not found at {baseline_dir}; generating it now.")
        voltage_path = generate_no_bess_voltages(case_path)
    else:
        print(f"Using existing voltage dataset: {baseline_dir}")

    column = f"{bus}_{BASELINE_PHASE}_pu"
    voltages = pd.read_csv(voltage_path, usecols=["timestamp", column])[column]
    if voltages.empty or not voltages.notna().all():
        raise ValueError(f"Voltage baseline {voltage_path} has no valid values for {column}")
    reference_voltage_pu = float(voltages.mean())
    power_base_kva = float(case.base.s_base_kva)
    gain_kw_per_pu = float(config.get("k", K))
    if gain_kw_per_pu < 0:
        raise ValueError("K must be nonnegative")

    strategy = VoltageProportionalStrategy(
        reference_voltage_pu,
        power_base_kva,
        gain_kw_per_pu,
    )
    return strategy, {
        "k_kw_per_pu": gain_kw_per_pu,
        "reference_voltage_pu": reference_voltage_pu,
        "reference_bus": bus,
        "reference_phase": BASELINE_PHASE,
        "baseline_steps": int(voltages.size),
        "baseline_path": str(Path(voltage_path)),
    }
