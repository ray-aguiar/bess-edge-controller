"""Run a selected rule-based BESS strategy in the OpenDSS environment."""
from importlib import import_module
import json
from pathlib import Path

from tools.evaluation import PolicyEvaluator
from tools.export_results import save_timeseries
from tools.plot_results import create_plots


ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "data" / "switzerland" / "5bus"
DEVICE = "b1"
ALGORITHM = "price_moving_average"
START_DATE = "2024-01-01"
END_DATE = "2024-01-10"
DAY = "2024-01-03"
SEED = 42
OUTPUT_DIR = Path("results/test/other/rule_based") / ALGORITHM


def main(output=OUTPUT_DIR, seed=SEED):
    algorithm = import_module(f"learning.algorithms.rule_based.{ALGORITHM}")
    config = {"case": str(CASE_PATH), "device": DEVICE,
              "window": getattr(algorithm, "WINDOW", 1),
              "extra_features": getattr(algorithm, "EXTRA_FEATURES", {}),
              "start_date": START_DATE, "end_date": END_DATE}
    strategy, algorithm_config = algorithm.create_strategy(config)
    evaluator = PolicyEvaluator(config, seed=seed)
    bess = next(device for device in evaluator.env.bess_list if device.id == DEVICE)
    try:
        records, summary = evaluator.evaluate(controller=strategy, record=True)
        base_kw = evaluator.env.normalizer.s_base_kva
        soc_values, requested_powers_kw, measured_powers_kw = [], [], []
        for record in records:
            info = record["info"]
            state = record["state"]
            soc_values.extend((state["device"]["soc_before_frac"],
                               info["executed_action"]["bess"][DEVICE]["soc_after_frac"]))
            requested_powers_kw.append(record["action"]["bess"][DEVICE]["p_net_pu"] * base_kw)
            measured_powers_kw.append(
                info["device_measurements"]["bess"][DEVICE]["p_net_total_pu"] * base_kw
            )
        soc_tolerance, power_tolerance = 1e-8, 1e-6
        if min(soc_values) < bess.soc_min_frac - soc_tolerance or max(soc_values) > bess.soc_max_frac + soc_tolerance:
            raise AssertionError("OpenDSS BESS state of charge exceeded its configured limits")
        if max(map(abs, measured_powers_kw)) > max(bess.p_charge_max_kw, bess.p_discharge_max_kw) + power_tolerance:
            raise AssertionError("OpenDSS BESS power exceeded its configured limit")

        output = Path(output)
        save_timeseries(records, output)
        report = {"config": evaluator.config, "seed": seed, "controller": ALGORITHM,
                  **algorithm_config, "pv_policy": "fixed",
                  "bess_limits": {"soc_min_frac": bess.soc_min_frac, "soc_max_frac": bess.soc_max_frac,
                                  "p_charge_max_kw": bess.p_charge_max_kw,
                                  "p_discharge_max_kw": bess.p_discharge_max_kw},
                  "requested_power_kw": {"min": min(requested_powers_kw), "max": max(requested_powers_kw)},
                  "measured_power_kw": {"min": min(measured_powers_kw), "max": max(measured_powers_kw)},
                  "soc_frac": {"min": min(soc_values), "max": max(soc_values)}, "summary": summary}
        (output / "records.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        evaluator.env.close()
    create_plots(input_dir=output, day=DAY)
    return records, summary

if __name__ == "__main__":
    main()
