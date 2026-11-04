import argparse
import json
import math
from pathlib import Path

import pandas as pd

from tools.scripts import compare_controllers as comparison
from opendss_environment import EnvironmentConfig, MicrogridEnv


def main(steps=comparison.STEPS, horizon=comparison.HORIZON, label=None):
    records = list(comparison.rollout(steps=steps, horizon=horizon))
    name = f"mpc-{steps}-h{horizon}" + (f"-{label}" if label else "")
    output = Path(__file__).resolve().parents[2] / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.json").write_text(json.dumps(records, default=str, indent=2), encoding="utf-8")
    rows = []

    for step, record in enumerate(records):
        state, action, info = record["state"], record["action"], record["info"]
        base = info["base"]["s_base_kva"]
        executed = info["executed_action"]
        bess = executed["bess"]["b1"]
        pv = executed["pv"]["pv1"]
        summary = record["summary"]
        action_errors, measurement_errors = [], []
        for kind, active, total in (("bess", "p_net_pu", "p_net_total_pu"), ("pv", "generation_pu", "generation_total_pu")):
            for device_id, command in action[kind].items():
                actual = executed[kind][device_id]
                measured = info["device_measurements"][kind][device_id]
                action_errors.extend((abs(command[active] - actual[total]) * base,
                                      abs(command["q_injection_pu"] - actual["q_injection_total_pu"]) * base))
                measurement_errors.extend((abs(measured[total] - actual[total]) * base,
                                           abs(measured["q_injection_total_pu"] - actual["q_injection_total_pu"]) * base))
        voltages = [value for bus in info["bus_voltages_pu"].values() for value in bus.values()]
        voltage_errors = [abs(predicted - value) for bus, predicted in record["predicted_voltages_pu"].items()
                          for value in info["bus_voltages_pu"][bus].values()]
        next_observation = record["next_observation"]
        timestamp_ok = pd.Timestamp(state["timestamp"]) == pd.Timestamp(info["timestamp"])
        feedback_error = 0.0
        if step + 1 < len(records):
            timestamp_ok = timestamp_ok and pd.Timestamp(next_observation["timestamp"]) == pd.Timestamp(state["timestamp"]) + pd.Timedelta(hours=state["dt_h"])
            feedback_error = records[step + 1]["state"]["bess"]["b1"]["soc_before_frac"] - bess["soc_after_frac"]
        else:
            timestamp_ok = timestamp_ok and next_observation is None
        rows.append({
            "timestamp": state["timestamp"], "horizon_objective_chf": summary["objective_cost"],
            "objective_energy_cost_error_chf": summary["objective_cost"] - summary["energy_cost"],
            "teacher_step_cost_chf": record["predicted_cost"], "environment_cost_chf": info["cost"],
            "reward": info["reward"], "reward_plus_cost": info["reward"] + info["cost"],
            "cost_error_chf": record["predicted_cost"] - info["cost"],
            "environment_cost_formula_error_chf": info["cost"] - (info["grid_import_pu"] * base * state["grid"]["buy_price_per_kwh"]
                                                                   - info["grid_export_pu"] * base * state["grid"]["sell_price_per_kwh"]) * state["dt_h"],
            "teacher_grid_import_kw": record["predicted_grid_kw"], "environment_grid_import_kw": -info["grid_p_pu"] * base,
            "grid_power_error_kw": record["predicted_grid_kw"] + info["grid_p_pu"] * base,
            "bess_executed_p_kw": bess["p_net_total_pu"] * base, "bess_executed_q_kvar": bess["q_injection_total_pu"] * base,
            "pv_available_kw": sum(info["observation_before"]["pv"]["pv1"]["available_pu"].values()) * base,
            "pv_executed_kw": pv["generation_total_pu"] * base, "pv_executed_q_kvar": pv["q_injection_total_pu"] * base,
            "soc_before_frac": bess["soc_before_frac"], "teacher_soc_after_frac": record["predicted_soc"]["b1"],
            "environment_soc_after_frac": bess["soc_after_frac"],
            "soc_prediction_error_frac": record["predicted_soc"]["b1"] - bess["soc_after_frac"],
            "soc_feedback_error_frac": feedback_error,
            "teacher_bess_loss_kw": record["predicted_bess_losses_kw"]["b1"],
            "environment_bess_loss_kw": bess["inverter_loss_pu"] * base,
            "bess_loss_prediction_error_w": (record["predicted_bess_losses_kw"]["b1"] - bess["inverter_loss_pu"] * base) * 1000,
            "simultaneous_charge_discharge_kw": min(record["predicted_bess_charge_kw"]["b1"], record["predicted_bess_discharge_kw"]["b1"]),
            "max_action_error_kw_kvar": max(action_errors), "max_measurement_error_kw_kvar": max(measurement_errors),
            "max_voltage_error_pu": max(voltage_errors), "environment_v_min_pu": min(voltages), "environment_v_max_pu": max(voltages),
            "socp_gap_max_normalized": summary["socp_gap_max_normalized"],
            "socp_current_error_max_a": summary["socp_current_error_max_a"], "socp_loss_error_max_w": summary["socp_loss_error_max_w"],
            "socp_tightness": summary["socp_tightness"], "solver_status": summary["status"],
            "valid_for_imitation": record["quality"]["valid_for_imitation"],
            "max_scaled_violation": record["quality"]["max_scaled_violation"],
            "loss_underestimate_max_w": record["quality"]["loss_underestimate_max_w"],
            "energy_execution_error_max_wh": record["quality"]["energy_execution_error_max_wh"],
            "timestamp_ok": timestamp_ok, "requested_action_ok": action == info["requested_action"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(output / "steps.csv", index=False)
    baseline = MicrogridEnv(EnvironmentConfig(comparison.CASE_PATH, episode_steps=steps))
    observation, _ = baseline.reset()
    baseline_reward = 0.0
    for step in range(steps):
        action = {
            "bess": {device_id: {"p_net_pu": 0.0, "q_injection_pu": 0.0} for device_id in observation["bess"]},
            "pv": {device_id: {"generation_pu": sum(values["available_pu"].values()), "q_injection_pu": 0.0}
                   for device_id, values in observation["pv"].items()},
        }
        observation, reward, terminated, truncated, info = baseline.step(action)
        baseline_reward += reward
        if terminated or truncated:
            break
    baseline.close()

    errors = ("objective_energy_cost_error_chf", "reward_plus_cost", "environment_cost_formula_error_chf",
              "cost_error_chf", "grid_power_error_kw", "soc_prediction_error_frac", "soc_feedback_error_frac",
              "max_action_error_kw_kvar", "max_measurement_error_kw_kvar", "max_voltage_error_pu",
              "simultaneous_charge_discharge_kw", "socp_gap_max_normalized", "socp_current_error_max_a", "socp_loss_error_max_w", "bess_loss_prediction_error_w")
    summary = {
        "steps": len(records), "horizon": horizon, "dt_h": records[0]["state"]["dt_h"],
        "solver_options": {},
        "base": records[0]["info"]["base"], "experiment_label": label,
        "solver": records[0]["quality"]["solver"]["name"],
        "valid_label_steps": int(table.valid_for_imitation.sum()),
        "max_scaled_violation": float(table.max_scaled_violation.max()),
        "loss_underestimate_max_w": float(table.loss_underestimate_max_w.max()),
        "energy_execution_error_max_wh": float(table.energy_execution_error_max_wh.max()),
        "teacher_first_step_cost_sum_chf": float(table.teacher_step_cost_chf.sum()),
        "environment_total_cost_chf": float(table.environment_cost_chf.sum()), "cumulative_reward": float(table.reward.sum()),
        "baseline_cost_chf": -baseline_reward, "baseline_savings_chf": float(-baseline_reward - table.environment_cost_chf.sum()),
        "baseline_description": "Same timestamps, idle BESS, full available PV, zero reactive commands; no terminal SoC adjustment",
        "soc_initial_frac": float(table.soc_before_frac.iloc[0]), "soc_final_frac": float(table.environment_soc_after_frac.iloc[-1]),
        "horizon_objective_min_chf": float(table.horizon_objective_chf.min()), "horizon_objective_max_chf": float(table.horizon_objective_chf.max()),
        "socp_tightness": {name: int(count) for name, count in table.socp_tightness.value_counts().items()},
        "solver_status": {name: int(count) for name, count in table.solver_status.value_counts().items()},
        "timestamp_mismatches": int((~table.timestamp_ok).sum()), "requested_action_mismatches": int((~table.requested_action_ok).sum()),
        "daylight_steps": int((table.pv_available_kw > 0).sum()), "pv_peak_kw": float(table.pv_available_kw.max()),
        "bess_active_power_max_abs_kw": float(table.bess_executed_p_kw.abs().max()),
        "voltage_min_pu": float(table.environment_v_min_pu.min()), "voltage_max_pu": float(table.environment_v_max_pu.max()),
        "maximum_absolute_errors": {column: float(table[column].abs().max()) for column in errors},
        "finite_numeric_results": all(math.isfinite(float(value)) for value in table.select_dtypes("number").to_numpy().ravel()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return table, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare teacher predictions with OpenDSS execution.")
    parser.add_argument("--steps", type=int, default=comparison.STEPS)
    parser.add_argument("--horizon", type=int, default=comparison.HORIZON)
    parser.add_argument("--label", help="Optional label to keep experiment outputs separate.")
    main(**vars(parser.parse_args()))
