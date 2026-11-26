"""Closed-loop evaluation of trained policies in the test environment."""
from collections import deque

import torch

from opendss_environment import EnvironmentConfig, MicrogridEnv
from opf_teacher.opf.single_phase import load_case
from tools.dataset import select_period
from tools.features import TARGETS, encode, local_observation


class PolicyEvaluator:
    """Reuse one environment, resetting its state before every policy evaluation."""

    def __init__(self, config, seed=0):
        case, period = select_period(load_case(config["case"]), config, default_steps=None)
        self.config = config | period
        self.seed = seed
        self.device = config["device"]
        device = next(device for device in case.bess if device.id == self.device)
        self.bus = case.buses[device.bus].name
        self.env = MicrogridEnv(EnvironmentConfig(case.config_path, episode_steps=period["steps"],
                                                start_step=period["start_step"]))

    def evaluate(self, model=None, record=False, disable_bess=False, controller=None):
        if not disable_bess:
            if (model is None) == (controller is None):
                raise ValueError("Provide exactly one model or controller when BESS control is enabled.")
            if model is not None:
                model.eval()
        if controller is not None:
            controller.reset()
        history = deque(maxlen=self.config["window"])
        observation, _ = self.env.reset(seed=self.seed)
        initial_soc = observation["bess"][self.device]["soc_before_frac"]
        bess = next(device for device in self.env.bess_list if device.id == self.device)
        p_charge_max_pu = bess.p_charge_max_kw / self.env.normalizer.s_base_kva
        p_discharge_max_pu = bess.p_discharge_max_kw / self.env.normalizer.s_base_kva
        records, total_reward, total_cost = [], 0.0, 0.0
        for step in range(self.config["steps"]):
            state = local_observation(observation, self.bus, self.device, self.config.get("extra_features"))
            history.append(state)
            window = [history[0]] * (history.maxlen - len(history)) + list(history)
            if disable_bess:
                predicted = torch.zeros(len(TARGETS))
            elif controller is not None:
                predicted = torch.tensor(controller.action(state, p_charge_max_pu, p_discharge_max_pu))
            else:
                with torch.inference_mode():
                    predicted = model(encode(window, self.config.get("features")).unsqueeze(0)).reshape(-1)
                if predicted.numel() != len(TARGETS):
                    raise ValueError("The model must return active and reactive BESS power")
            action = {
                "bess": {key: {"p_net_pu": 0.0, "q_injection_pu": 0.0} for key in observation["bess"]} |
                        {self.device: dict(zip(TARGETS, predicted.tolist()))},
                "pv": {key: {"generation_pu": sum(values["available_pu"].values()), "q_injection_pu": 0.0}
                       for key, values in observation["pv"].items()},
            }
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            total_cost += info["cost"]
            if record:
                records.append({"state": state, "action": action, "info": info})
            if terminated or truncated:
                break
        return records, {"steps": step + 1, "cost_chf": total_cost, "reward": total_reward,
                         "soc_initial_frac": initial_soc,
                         "soc_final_frac": info["executed_action"]["bess"][self.device]["soc_after_frac"]}
