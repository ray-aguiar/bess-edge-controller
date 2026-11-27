# Training and evaluation guide

Imitation learning for local battery energy storage control using PyTorch,
OPF teacher actions, and OpenDSS simulations. Each neural network receives
only measurements from its device's bus and the device's own state, and
outputs commands for that device. The teacher solves the full network OPF
to provide reference actions.

The project includes two editable Git submodules:

- [opf_teacher](https://github.com/TeradaZenichi/opf-teacher): reference actions from optimal power flow.
- [opendss_environment](https://github.com/TeradaZenichi/powerflow-opendss-environment): electrical simulation through a Gymnasium environment.

Research scope, formulation and experiment records are in [docs/](README.md).
The root Python entry points include `train.py`, `test.py`, `test_rule_based.py`
and `test_no_bess.py`. Essential helpers live in `tools/`; standalone
verification and profile-generation scripts live in `tools/scripts/`. MLP,
LSTM and supervised behavior cloning are implemented, as is the
`price_moving_average` rule-based baseline.

Run commands from the repository root. The [main README](../README.md) provides the quick start.

## Installation

Requires Python 3.11+ and Git LFS. Continuous SOCP cases use Clarabel on CPU;
cases with SOS controls require a valid Gurobi license.

```powershell
git clone --recurse-submodules https://github.com/ray-aguiar/bess-edge-controller.git
cd bess-edge-controller
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an existing checkout, run `git submodule update --init --recursive` first.
Dependencies include PyTorch for CPU; both submodules are installed in editable mode.

## Select dataset dates

The dataset configuration in `train.py` and `test.py` accepts `start_date` (inclusive) and
`end_date` (exclusive) in their configuration. Dates without an offset use
`Europe/Zurich` when the dataset has timezone-aware timestamps; explicit
UTC/offset dates also work.
For example, `2023-01-01` to `2023-01-02` selects one Swiss calendar day.
Boundaries must match dataset intervals and remain within the available data.
The selected range determines the number of steps.
Daylight-saving days may contain 92 or 100 intervals.

Without dates, collection and policy testing use the complete dataset.
Each selected range starts a fresh SoC/history episode.
The saved configuration includes resolved dates, row offset and step count.

## Generate and reuse teacher data

```powershell
.\.venv\Scripts\python.exe train.py
```

Edit `DATASET_CONFIG` in `train.py` for the case, device, dates, window,
OPF block size and `extra_features`. Use `"extra_features": {}` to disable
optional inputs. Training loads a complete matching dataset if it exists;
otherwise it runs the generator. Generated datasets contain `x.pt`, `y.pt`,
`config.json` and inspectable `x.json`/`y.json` under
`results/datasets/<case_name>/<dataset_name>/`. The dataset folder name uses
the starting year, device, window, feature mode (`extra` or `no_extra`) and an
eight-character hash of the full configuration. For example:
`results/datasets/switzerland_5bus/2023_b1_w96_extra_a1b2c3d4/`.
The hash distinguishes different dates, feature selections, cases and OPF
settings without making the folder name long. The current `train.py`
configuration uses a 96-step window and the normalized grid buy price as its
only extra input, in addition to the six base inputs.

```python
from tools.generate_dataset import generate_dataset

config = {"window": 3, "device": "b1", "case": "data/switzerland/5bus",
          "extra_features": {},
          "block_weeks": None,
          "start_date": "2023-01-01", "end_date": "2023-01-02"}
dataset_dir = generate_dataset(config)
```

Collection uses `opf_teacher.collect_pairs(case, PairConfig(...))`; the teacher
solves the selected period and owns local selection and history assembly. This repository
selects the dates, fixes PV operation and converts the returned pairs to tensors.
The teacher reconstructs observations using the optimized SoC and previous actions;
no environment is stepped during collection. Dates define the complete optimization
period, so profiles never enter the next dataset partition. The saved configuration
records `teacher_mode="weekly_blocks"` when `block_weeks` is set and
`"single_optimization"` when it is `None`. X contains the selected
bus and battery signals plus requested extras; Y is that battery's current P/Q
action in pu. Collection sets `pad_initial=True`:
missing past observations repeat the first state, matching policy evaluation.
With window 3, the first windows are `[x0, x0, x0]`, `[x0, x0, x1]`, then
`[x0, x1, x2]`; 96 valid steps produce 96 pairs. This initialization occurs only
at episode start, not at weekly block boundaries. Voltage is calculated before the new action, using current
loads and held previous device powers (PV is capped by current availability).
A radial power-flow snapshot supplies it without stepping the environment or
integrating battery energy. It never uses the new action's resulting voltage.

Set `block_weeks=1` to optimize consecutive one-week blocks, or another positive
integer for longer blocks. `None` keeps one optimization over the entire selected
period. A week means 168 elapsed hours (672 intervals at 15 minutes); the last
block may be shorter. SoC, previous BESS/PV commands and observation history carry
across blocks. The original terminal SoC target applies only to the last block.
The saved mode becomes `teacher_mode="weekly_blocks"`. This reduces optimization
memory, but does not guarantee the optimum of a single full-period problem.

X is a CPU float32 tensor with shape `[samples, window, F]`, and Y has shape
`[samples, 2]`. Feature order is stored in the dataset `config.json` and reused
by `test.py` through the resulting checkpoint metadata. The same dataset can be
used to train both network architectures without running OPF collection again.

```python
import torch
from tools.generate_dataset import dataset_directory
from train import DATASET_CONFIG

dataset_dir = dataset_directory(DATASET_CONFIG)
x = torch.load(dataset_dir / "x.pt", map_location="cpu", weights_only=True)
y = torch.load(dataset_dir / "y.pt", map_location="cpu", weights_only=True)
```

The original pairs remain in `x.json` and `y.json` for inspection, including
observation timestamps. The dataset `config.json` records generation settings,
resolved dates and feature/target order. Training randomness is configured
separately in `train.py`.

## Train a behavior-cloning policy

Choose `ARCHITECTURE`, training hyperparameters, evaluation fraction and seed
in `train.py`, then run:

```powershell
.\.venv\Scripts\python.exe train.py
```

Training loads `x.pt` and `y.pt` (generating them first only if needed), makes
the seeded train/evaluation split, and saves model checkpoints under
`results/train/`. Current defaults use an MLP, seed 42 and the price extra
feature; the best checkpoint is saved to
`results/train/supervised/mlp/<dataset_name>/seed_42/trial_0/best.pt`.
Change `ARCHITECTURE` to `"lstm"` to train an LSTM from the same dataset;
teacher data does not need to be regenerated. The checkpoint's
`training.json` keeps the dataset features and configuration required by
`test.py`.

Optional `extra_features` in `train.py` adds pre-action measurements
and communicated device signals without changing the controlled BESS or its P/Q targets:

```python
"extra_features": {
    "time": ["hour_sin", "hour_cos", "month_sin", "month_cos"],
    "branches": ["p_flow_pu", "q_flow_pu"],
    "bess": {"b2": ["soc_before_frac", "previous_p_pu"]},
    "pv": {"pv1": ["available_pu", "previous_p_pu"]},
    "grid": ["buy_price_01"],
    "bus": ["angle_before_deg"],
}
```

The `time` group accepts any subset of `hour_sin`, `hour_cos`, `month_sin`
and `month_cos`; it computes circular encodings from each observation's
timestamp. Hours use local timestamp time with sub-hour precision, and months
use the 1-based calendar month. Because these are ordinary selected extras,
the same configuration adds them to teacher data and live policy inputs.
`grid.buy_price_01` is the buy tariff scaled with the same fixed bounds across
the standard, time-of-use and dynamic profiles:
`(buy_price_per_kwh - 0.14672) / (0.288823 - 0.14672)`. All prices in those
profiles fall within these bounds; out-of-range prices raise an error rather
than silently changing or clipping the scaling.

Use only device IDs present in the case: the supplied Swiss case has `b1` and
`pv1`, but no `b2`. Omit groups you do not need, or use `{}` for the original six inputs.
Nonempty dataset `extra_features` automatically sets the metadata label
`"extra_features"`; with `{}`, the label is unset. The label and feature
selection are saved with the dataset and copied into `training.json`.
Dataset-specific output folders distinguish feature selections by a short
configuration hash.
Branch signals include every terminal connected to the controlled BESS bus,
identified by element, terminal and phase; optional fields also include `current_a`
and `current_angle_deg`. Power is positive entering the line/transformer terminal,
following OpenDSS. Current is in amperes and angles are in degrees; these inputs
are not standardized automatically. Teacher branch extras currently support the
single-phase equivalent; communicated device signals retain phase names.

Communication is instantaneous and uses current SoC/availability and previous
device powers, never simultaneous new actions. `config.json` and `training.json`
save `extra_features` and the complete ordered input paths in `config.features`.
The six default inputs come first; extra paths are sorted by group, identifier and field.
`test.py` restores both from the training configuration, including for older
checkpoints without extras. The test evaluator controls the selected BESS,
keeps other BESS commands at P=Q=0, and uses available PV with Q=0.

Set `ARCHITECTURE` in `train.py` to `"mlp"` (default) or `"lstm"`. The LSTM reads
`[batch, window, features]` in chronological order and uses its last output
to predict the current BESS P/Q action. It has one recurrent layer; `hidden_size`
sets its width. Its hidden state starts at zero for each window, so memory is
limited to the supplied history. The MLP flattens the same window.

`train.py` performs this split and training loop from the selected dataset tensors.
Each observation window keeps its original order. Each seed has its own
`seed_42/`, `seed_43/`, etc. under the model folder; each iteration writes
weights and history to `trial_0/`, `trial_1/`, etc.
The class creates the selected CPU network, trains with Adam and MSE, and returns the lowest-MSE
model in evaluation mode. Training hyperparameters come from the `Hyperparameter` object;
`config` describes the dataset and is recorded for reproducibility.
`weights.pt` stores the final model; `best.pt` stores the lowest-validation-MSE
model. `training.json` records the best epoch and training/validation MSE history.
Training and checkpoint selection do not run the environment or use reward.

For five repetitions, use seeds such as `[42, 43, 44, 45, 46]` with identical
data, hyperparameters and evaluation dates. Pass each run's seed to `train()`
and `test.main()`, using separate output folders such as `results/seed_42/`.
The DataLoader has its own seeded generator, and policy testing passes the
supplied seed to the environment's `reset()`. Teacher data and
the current environment scenarios are deterministic, so changing the seed
changes training randomness while preserving the scenario. Reproducibility
assumes the same code, data, dependency versions and execution platform.

The evaluation fraction is a seeded random split over windows, so nearby
windows can share historical observations across the train/evaluation split.
For stronger temporal generalization, evaluate on dates outside the training
period, as the default test period does.
Hyperparameters are selected manually for now; a future Bayesian search can
supply a new `Hyperparameter` object per trial and compare validation MSE.
`torch.nn.Module.train()` only switches the network's mode; the algorithm's
`train()` method performs the optimization.

## Test a trained policy

`test.py` loads an MLP or LSTM checkpoint and runs it in OpenDSS between the selected
dates. Set `config["architecture"]` manually to `"mlp"` or `"lstm"`, matching the
checkpoint. It reads the architecture dimensions, observation window and device ID
from `training.json` next to the checkpoint. The test main block selects the
architecture and `MODEL_EXTRA_FEATURES`; set both to match the training run.
Its default selection is MLP without extras, so set
`MODEL_EXTRA_FEATURES = {"grid": ["buy_price_01"]}` for a model trained with
the current `train.py` defaults. `main()` accepts a single checkpoint. Set the model path, dates, output
folder and seed in its main block, or call:

```python
from test import main

records, summary = main(model_path="results/train/supervised/lstm/seed_42/trial_0/best.pt",
                        config={"case": "data/switzerland/5bus", "architecture": "lstm",
                                "start_date": "2024-02-29", "end_date": "2024-03-01"},
                        output="results/test/seed_42/trial_0", seed=42)
```

The CPU `torch.nn.Module` receives float32 input `[batch, window, F]` and
returns `[batch, 2]` in pu. Input features, in order, are current pre-action
voltage, current local P/Q demand, current SoC, and previous executed
BESS P/Q. Outputs are charging-positive active power and injection-positive
reactive power. Both networks use the same encoding as training.
The first observation is repeated to fill the NN window at episode startup.
Testing controls one selected BESS and reads its actual SoC and measured
previous controls from the environment at every step. Other connected BESS
receive zero P/Q commands. The current encoder selects phase a; three-phase
BESS policy experiments require a suitable encoder and policy, even though
the environment and teacher libraries also support three-phase cases.

Datasets and checkpoints record `voltage_source = "pre_action"`. Regenerate
data and retrain old previous-interval-voltage models with `train.py` before
using the updated `test.py`; older checkpoints are rejected explicitly.

Collection and policy testing both use available PV generation with Q=0; the
teacher optimizes only the BESS. This fixed policy is recorded in the metadata.
The fixed PV policy is applied in memory. The current scenario uses
`sell_price_per_kwh = 0.2 * price_per_kwh` and `feed_in_tariff_ratio=0.2`;
older symmetric-tariff experiments used ratio 1.0.

Results include `records.json` with local states, requested actions and executed
environment results, plus `summary.json` with the checkpoint path, dates, seed,
executed cost and reward. There is no terminal-energy cost adjustment. Generated artifacts stay in the
Git-ignored `results/` folder. Exploratory single-seed results do not establish general performance or safety.

`buses.csv` and `branches.csv` use one row per timestamp and one column per
measurement. Device CSVs use one row per interval and device:

| File             | Measurements                                                             |
| ---------------- | ------------------------------------------------------------------------ |
| `buses.csv`    | Voltage in V/pu, angle, and net P/Q injection into connected branches    |
| `branches.csv` | P/Q, current in A and angle for each line/transformer terminal and phase |
| `bess.csv`     | Requested and measured P/Q, SoC before/after the interval                |
| `pv.csv`       | Available and measured generation, reactive injection                    |
| `grid.csv`     | Imported/exported power, reactive injection, cost and reward             |

For the single-phase case, columns include `bus_001_v_pu`,
`bus_004_p_injection_kw`, `line_line_001_002_t1_p_kw` and
`line_line_001_002_t2_current_a`. The branch prefix combines its OpenDSS
class/name and terminal number. Multiphase circuits also include the phase,
for example `source_a_v_pu` and `line_feeder_t1_a_current_a`.

Every row has `timestamp`, identifying the start of the interval. Measurements
refer to the executed action, before preparation of the next interval;
`soc_after_frac` refers to the end of that interval. Powers are kW/kvar and SoC is
a fraction. Branch P/Q is positive entering the element from the listed bus;
both terminal directions are saved. BESS P is positive when charging and Q
is positive for injection. CSV export lives in `tools/export_results.py`.

The no-BESS baseline is available with:

```powershell
.\.venv\Scripts\python.exe test_no_bess.py
```

It runs the same fixed-PV OpenDSS evaluation with BESS commands disabled and
writes results to `results/test/other/no_bess/`.

For a simple rule-based baseline, run:

```powershell
.\.venv\Scripts\python.exe test_rule_based.py
```

The script selects `price_moving_average` by default. Its strategy uses a
96-interval (24-hour) moving average, initialized from the prices immediately
before the test period, and charges or discharges at 20% of the configured
power limit when the current price is below or above that average. It ignores
PV output and grid import/export state; reactive power is zero. Change
`WINDOW` and `POWER_FRACTION` in
`learning/algorithms/rule_based/price_moving_average.py` to adjust the rule.
Results, including requested/measured BESS power, SoC, cost and grid flows,
are saved to `results/test/other/rule_based/price_moving_average/`; the script
checks the configured SoC and power limits.

The previous MPC/NN comparison is preserved in
`tools/scripts/compare_controllers.py`. The detailed teacher/environment check
remains available:

```powershell
.\.venv\Scripts\python.exe -m tools.scripts.evaluate_mpc --steps 96 --horizon 8 --label contract
```

## Working on the submodules

Edit either submodule directly and rerun the example. From a fresh clone,
create a work branch inside the submodule before committing changes.
Push submodule commits first, then commit their updated references here.
Use `git push --recurse-submodules=check` when publishing this project.
