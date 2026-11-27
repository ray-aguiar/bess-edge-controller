import json
from pathlib import Path

from tools.evaluation import PolicyEvaluator
from tools.export_results import save_timeseries
from tools.plot_results import create_plots

CASE_PATH = Path(__file__).resolve().parent / "data" / "switzerland" / "5bus" #"5bus-switzerland"
DEVICE = "b1"
START_DATE = "2024-01-01"
END_DATE = "2024-01-10"
DAY = "2024-01-03"  # Day used for daily plots.
OUTPUT_DIR = Path("results/test/other/no_bess")

def main(config, output=OUTPUT_DIR):
    evaluator = PolicyEvaluator(config, seed=None)
    records, summary = evaluator.evaluate(model=None, record=True, disable_bess=True)
    evaluator.env.close()

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_timeseries(records, output)

    report = {"config": evaluator.config, "pv_policy": "fixed",
              "bess_policy": "disabled", "summary": summary}

    (output / "records.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return records, summary


if __name__ == "__main__":
    config = {"case": str(CASE_PATH), "device": DEVICE, "window": 1,
              "extra_features": {}, "start_date": START_DATE, "end_date": END_DATE}
    records, summary = main(config)
    create_plots(input_dir=OUTPUT_DIR, day=DAY)