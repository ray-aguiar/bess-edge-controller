from pathlib import Path

from opendss_environment import EnvironmentConfig, MicrogridEnv
from opf_teacher.opf.single_phase import load_case
from opf_teacher.teacher import Teacher


CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"
HORIZON = 8
STEPS = 96


def rollout(steps=STEPS, horizon=HORIZON, case=None):
    """Run a fixed forecast horizon, or the remaining episode when horizon is None."""
    periods = steps if horizon is None else steps + horizon - 1
    case = (load_case(CASE_PATH) if case is None else case).horizon(0, periods=periods)
    env = MicrogridEnv(EnvironmentConfig(case.config_path, episode_steps=steps))
    observation, _ = env.reset()

    for step in range(steps):
        teacher = Teacher(case.horizon(step, periods=horizon), formulation="single_phase_socp")
        teacher.observe_dict(observation)
        solved = teacher.solve()
        state, action = teacher.state_action()
        observation, reward, terminated, truncated, info = env.step(action)

        record = {
            "state": state,
            "action": action,
            "info": info,
            "next_observation": observation,
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
        }
        print(step, state["timestamp"], "SOCP:", solved.summary.socp_tightness,
              "horizon objective:", solved.summary.objective_cost,
              "step cost:", record["predicted_cost"], "reward:", reward, flush=True)
        yield record
        if terminated or truncated:
            break

    env.close()


def main(steps=STEPS, horizon=HORIZON, case=None):
    """Return the complete trajectory for the existing MPC API."""
    return list(rollout(steps=steps, horizon=horizon, case=case))


if __name__ == "__main__":
    print("MPC steps:", len(main()))
