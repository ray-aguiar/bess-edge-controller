from collections import deque
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from opendss_environment import EnvironmentConfig, MicrogridEnv
from opf_teacher.opf.single_phase import load_case
from opf_teacher.teacher import Teacher

from tools.features import TARGETS, encode, local_observation
from tools.dataset import configure_pv, select_period


CASE_PATH = Path(__file__).resolve().parents[2] / "data" / "5bus-switzerland"
HORIZON = 8
STEPS = 96


def rollout(steps=STEPS, horizon=HORIZON, case=None, *, model=None, config=None, seed=0):
    """Run one controller in its own OpenDSS environment."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = {"window": 3, "device": "b1", "pv_policy": "teacher"} | (config or {})
    periods = steps if horizon is None else steps + horizon - 1
    case = load_case(CASE_PATH) if case is None else case
    case = case.horizon(0, periods=min(periods, case.n_periods))
    configure_pv(case, config["pv_policy"])
    history = deque(maxlen=config["window"])
    if model is not None:
        model.cpu().eval()
        if len(case.bess) != 1 or case.bess[0].id != config["device"]:
            raise ValueError("NN comparison currently requires one BESS matching 'device'")
        bus = case.buses[case.bess[0].bus].name
    env = MicrogridEnv(EnvironmentConfig(case.config_path, episode_steps=steps,
                                        start_step=config.get("start_step", 0)))
    observation, _ = env.reset(seed=seed)

    for step in range(steps):
        started = perf_counter()
        solved = None
        if model is None:
            forecast = None if horizon is None else min(horizon, case.n_periods - step)
            teacher = Teacher(case.horizon(step, periods=forecast), formulation="single_phase_socp")
            teacher.observe_dict(observation)
            solved = teacher.solve()
            state, action = teacher.state_action()
        else:
            action = {
                "bess": {config["device"]: {}},
                "pv": {key: {"generation_pu": sum(values["available_pu"].values()), "q_injection_pu": 0.0}
                       for key, values in observation["pv"].items()},
            }
        if model is not None:
            state = local_observation(observation, bus, config["device"], config.get("extra_features"))
            history.append(state)
            window = [history[0]] * (history.maxlen - len(history)) + list(history)
            with torch.inference_mode():
                predicted = model(encode(window, config.get("features")).unsqueeze(0)).reshape(-1)
            if predicted.numel() != 2:
                raise ValueError("The NN must return two pu outputs: active and reactive BESS power")
            action["bess"][config["device"]] = dict(zip(TARGETS, predicted.tolist()))
        decision_seconds = perf_counter() - started
        observation, reward, terminated, truncated, info = env.step(action)

        record = {
            "state": state,
            "action": action,
            "info": info,
            "next_observation": observation,
            "terminated": terminated, "truncated": truncated,
            "decision_seconds": decision_seconds,
            "controller": "mpc" if model is None else "nn",
        }
        if solved is not None and model is None:
            record.update({
                "predicted_soc": {device.id: float(device.result.soc_frac.iloc[0]) for device in solved.bess},
                "predicted_cost": float(solved.grid.result.cost.iloc[0]),
                "predicted_grid_kw": float(solved.grid.result.import_kw.iloc[0] - solved.grid.result.export_kw.iloc[0]),
                "predicted_voltages_pu": {bus.name: float(bus.result.v_pu.iloc[0]) for bus in solved.buses.values()},
                "predicted_bess_losses_kw": {device.id: float(device.result.inverter_loss_kw.iloc[0]) for device in solved.bess},
                "predicted_bess_charge_kw": {device.id: float(device.result.charge_kw.iloc[0]) for device in solved.bess},
                "predicted_bess_discharge_kw": {device.id: float(device.result.discharge_kw.iloc[0]) for device in solved.bess},
                "summary": solved.summary.as_dict(),
                "quality": solved.quality,
                "socp_tightness": solved.summary.socp_tightness,
            })
        yield record
        if terminated or truncated:
            break

    env.close()


def main(model=None, config=None, output="results/comparison", seed=0):
    """Compare MPC and an optional trained CPU model on independent trajectories."""
    config = {"case": str(CASE_PATH), "steps": STEPS, "horizon": HORIZON,
              "window": 3, "device": "b1", "start_date": None, "end_date": None} | (config or {})
    if config["window"] < 1:
        raise ValueError("window must be positive")
    case, period = select_period(load_case(config["case"]), config, config["steps"])
    config |= period
    controllers = {"mpc": None} if model is None else {"mpc": None, "nn": model}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, summary = {}, {}
    for name, controller in controllers.items():
        records[name] = list(rollout(config["steps"], config["horizon"], case,
                                    model=controller, config=config | {"pv_policy": "fixed"}, seed=seed))
        run = records[name]
        voltages = [value for record in run for bus in record["info"]["bus_voltages_pu"].values()
                    for value in bus.values()]
        summary[name] = {
            "steps": len(run), "cost_chf": sum(record["info"]["cost"] for record in run),
            "reward": sum(record["info"]["reward"] for record in run),
            "soc_initial_frac": case.bess[0].soc_init_frac,
            "soc_final_frac": run[-1]["info"]["executed_action"]["bess"][config["device"]]["soc_after_frac"],
            "voltage_min_pu": min(voltages), "voltage_max_pu": max(voltages),
            "decision_seconds_mean": sum(record["decision_seconds"] for record in run) / len(run),
        }
        (output / f"{name}.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    report = {"config": config, "seed": seed, "pv_policy": "fixed",
              "terminal_energy_adjusted": False, "summary": summary}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return records, summary


if __name__ == "__main__":
    config = {"case": str(CASE_PATH), "steps": 96, "horizon": 8,
              "window": 3, "device": "b1", "start_date": "2023-01-01", "end_date": "2023-01-02"}
    model = None  # Set a trained torch.nn.Module, including its preprocessing, here.
    seed = 42
    records, summary = main(model, config, output="results/comparison", seed=seed)
