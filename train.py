from pathlib import Path

import mpc
from opf_teacher.opf.single_phase import load_case
from opf_teacher.teacher import PairSelector, StateActionHistory


CASE_PATH = Path(__file__).resolve().parent / "data" / "5bus-switzerland"
HORIZON = 96
WINDOW = 3
DEVICE_ID = "b1"


def main():
    case = load_case(CASE_PATH).horizon(0, periods=HORIZON)
    selector = PairSelector(case, mode="local", device_id=DEVICE_ID)
    history = StateActionHistory(window=WINDOW)
    pairs = []

    for record in mpc.rollout(steps=HORIZON, horizon=None, case=case):
        local_state, local_action = selector.select(record["state"], record["action"])
        pair = history.append(local_state, local_action)

        if pair is not None and record["quality"]["valid_for_imitation"]:
            pairs.append(pair)

    print("Pairs:", len(pairs), "Window:", WINDOW)
    return pairs


if __name__ == "__main__":
    main()
