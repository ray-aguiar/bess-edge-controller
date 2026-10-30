from pathlib import Path

from opendss_environment import EnvironmentConfig, MicrogridEnv
from opf_teacher.opf.single_phase import load_case
from opf_teacher.teacher import PairSelector, StateActionHistory, Teacher


CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"
HORIZON = 96
WINDOW = 3
DEVICE_ID = "b1"


def main():
    case = load_case(CASE_PATH).horizon(0, periods=HORIZON)
    selector = PairSelector(case, mode="local", device_id=DEVICE_ID)
    history = StateActionHistory(window=WINDOW)
    env = MicrogridEnv(EnvironmentConfig(CASE_PATH, HORIZON))
    pairs = []

    observation, _ = env.reset()
    for step in case.periods:
        teacher = Teacher(case.horizon(step), formulation="single_phase_socp")
        teacher.observe_dict(observation)
        solved = teacher.solve()
        state, action = teacher.state_action()
        local_state, local_action = selector.select(state, action)
        pair = history.append(local_state, local_action)

        if pair is not None:
            pairs.append(pair)

        print("Step:", step, "SOCP:", solved.summary.socp_tightness)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

    print("Pairs:", len(pairs), "Window:", WINDOW)
    return pairs


if __name__ == "__main__":
    main()
