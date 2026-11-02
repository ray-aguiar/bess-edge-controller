"""Shared local feature order for offline data and CPU policy evaluation."""
import torch


FEATURES = (
    ("bus", "v_before_pu"), ("bus", "p_load_pu"), ("bus", "q_load_pu"),
    ("device", "soc_before_frac"), ("device", "previous_p_pu"), ("device", "previous_q_pu"),
)
TARGETS = ("p_net_pu", "q_injection_pu")


def encode(history):
    """Convert one observation window to float32 [window, 6] on CPU."""
    return torch.tensor([[state[group][name] for group, name in FEATURES] for state in history],
                        dtype=torch.float32)


def local_observation(observation, bus, device, voltage):
    measured_bus = observation["buses"][bus]
    measured_device = observation["bess"][device]
    return {
        "timestamp": observation["timestamp"], "dt_h": observation["dt_h"],
        "bus": {"v_before_pu": voltage, "p_load_pu": measured_bus["p_load_pu"]["a"],
                "q_load_pu": measured_bus["q_load_pu"]["a"]},
        "device": {"soc_before_frac": measured_device["soc_before_frac"],
                   "previous_p_pu": measured_device["previous_p_pu"]["a"],
                   "previous_q_pu": measured_device["previous_q_pu"]["a"]},
    }
