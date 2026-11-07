"""Shared local feature order for offline data and CPU policy evaluation."""
import torch
from opf_teacher.teacher import select_extra_features


FEATURES = (
    ("bus", "v_before_pu"), ("bus", "p_load_pu"), ("bus", "q_load_pu"),
    ("device", "soc_before_frac"), ("device", "previous_p_pu"), ("device", "previous_q_pu"),
)
TARGETS = ("p_net_pu", "q_injection_pu")


def feature_order(state):
    """Keep default inputs first, followed by explicitly selected extra signals."""
    def paths(value, prefix):
        if isinstance(value, dict):
            return [path for key in sorted(value) for path in paths(value[key], prefix + (key,))]
        return [prefix]
    return list(FEATURES) + paths(state["extras"], ("extras",)) if state.get("extras") else list(FEATURES)


def encode(history, features=None):
    """Convert a chronological observation window to CPU float32."""
    features = feature_order(history[0]) if features is None else features
    rows = []
    for state in history:
        row = []
        for path in features:
            value = state
            for key in path:
                value = value[key]
            row.append(value)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def local_observation(observation, bus, device, extra_features=None):
    measured_bus = observation["buses"][bus]
    measured_device = observation["bess"][device]
    state = {
        "timestamp": observation["timestamp"], "dt_h": observation["dt_h"],
        "bus": {name: measured_bus[name]["a"] for name in ("v_before_pu", "p_load_pu", "q_load_pu")},
        "device": {"soc_before_frac": measured_device["soc_before_frac"]} |
                  {name: measured_device[name]["a"] for name in ("previous_p_pu", "previous_q_pu")},
    }
    if extra_features:
        state["extras"] = select_extra_features(observation, bus, extra_features)
    return state
