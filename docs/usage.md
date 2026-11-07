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
The root Python entry points are `train.py` and `test.py`. Essential helpers
live in `tools/`; standalone verification and profile-generation scripts live
in `tools/scripts/`. Network architectures live in `learning/networks/` and
learning algorithms in `learning/algorithms/`. MLP, LSTM and supervised behavior cloning
are implemented; other architectures and algorithms remain empty scaffolds.

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

Both scripts accept `start_date` (inclusive) and `end_date` (exclusive) in the
configuration dictionary. Dates without an offset use `Europe/Zurich` when the
dataset has timezone-aware timestamps; explicit UTC/offset dates also work.
For example, `2023-01-01` to `2023-01-02` selects one Swiss calendar day.
Boundaries must match dataset intervals and remain within the available data.
The selected range determines the number of steps.
Daylight-saving days may contain 92 or 100 intervals.

Without dates, collection and policy testing use the complete dataset.
Each selected range starts a fresh SoC/history episode.
The saved configuration includes resolved dates, row offset and step count.

## Collect teacher data

```powershell
.\.venv\Scripts\python.exe train.py
```

The main block collects tensors once and trains BehaviorCloning in a loop with
`max_iterations = 1`, saving the model under `results/train/supervised/lstm/extra_features/seed_42/trial_0/` with the current defaults.
It uses the selected 2023 range for an 80/20 random train/evaluation split (`eval_fraction=0.2`),
with the supplied seed. `x_train`/`y_train` feed training; `x_eval`/`y_eval`
are evaluated after every epoch. The test period is selected separately in 2024.
Edit `config`, `hyperparameters`, `output` and `seed` in that block,
or call the collector directly:

```python
from train import main

config = {"window": 3, "device": "b1", "case": "data/5bus-switzerland",
          "block_weeks": None,
          "start_date": "2023-01-01", "end_date": "2023-01-02"}
x, y = main(config, output="results/train/seed_42", seed=42)
```

Collection uses `opf_teacher.collect_pairs(case, PairConfig(...))`; the teacher
solves the selected period once and owns local selection and history assembly. This repository
selects the dates, fixes PV operation and converts the returned pairs to tensors.
The teacher reconstructs observations using the optimized SoC and previous actions;
no environment is stepped during collection. Dates define the complete optimization
period, so profiles never enter the next dataset partition. The saved configuration
records `teacher_mode="single_optimization"` by default. X contains only the selected bus and battery signals;
Y is that battery's current P/Q action in pu. Training sets `pad_initial=True`:
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

The function returns CPU float32 PyTorch tensors: X has shape
`[samples, window, F]` and Y has shape `[samples, 2]`, where F is six base features plus the selected extras. The feature order is shared
with `test.py`. For 96 valid intervals and window 3, their shapes are
`[96, 3, 6]` and `[96, 2]`. The tensors are saved as `x.pt` and `y.pt`:

```python
import torch

x = torch.load("results/train/seed_42/x.pt")
y = torch.load("results/train/seed_42/y.pt")
```

The original pairs remain in `x.json` and `y.json` for inspection, including
observation timestamps. `config.json` records the seed, settings and
feature/target order. Python, NumPy and PyTorch share the supplied seed;
PyTorch deterministic algorithms are enabled, and profiles stay chronological.
`main()` prepares and returns X/Y; training is coordinated outside that function.

## Train a behavior-cloning policy

Optional `extra_features` in the training configuration adds pre-action measurements
and communicated device signals without changing the controlled BESS or its P/Q targets:

```python
"extra_features": {
    "branches": ["p_flow_pu", "q_flow_pu"],
    "bess": {"b2": ["soc_before_frac", "previous_p_pu"]},
    "pv": {"pv1": ["available_pu", "previous_p_pu"]},
    "grid": ["buy_price_per_kwh", "sell_price_per_kwh"],
    "bus": ["angle_before_deg"],
}
```

Use only device IDs present in the case: the supplied Swiss case has `b1` and
`pv1`, but no `b2`. Omit groups you do not need, or use `{}` for the original six inputs.
Nonempty `extra_features` automatically selects the label `"extra_features"`:
`results/train/supervised/lstm/extra_features/seed_42/trial_0/`.
With `extra_features={}`, the original folder structure is preserved.
The label is saved in the configuration; set `label="extra_features"` in
the `test.py` main block to select the experiment. Existing `local_p_flow`
experiments keep their original paths. Test outputs keep the same hierarchy.
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

Set `config["architecture"]` to `"lstm"` (default) or `"mlp"`. The LSTM reads
`[batch, window, features]` in chronological order and uses its last output
to predict the current BESS P/Q action. It has one recurrent layer; `hidden_size`
sets its width. Its hidden state starts at zero for each window, so memory is
limited to the supplied history. The MLP flattens the same window.

Using the training tensors prepared by the main block, create the trainer and run the loop:

```python
from pathlib import Path
from learning.algorithms.behavior_cloning import BehaviorCloning, Hyperparameter

hyperparameters = Hyperparameter(hidden_size=64, learning_rate=1e-3,
                                batch_size=64, epochs=100)
seed = 42
trainer = BehaviorCloning(hyperparameters)
output = Path("results/train") / f"seed_{seed}"
max_iterations = 1
for iteration in range(max_iterations):
    trained_model = trainer.train(x_train, y_train, x_eval, y_eval, config,
                                  output=output / f"trial_{iteration}", seed=seed)
```

The `if __name__ == "__main__"` block collects data once by calling `main()`,
splits sample indices with `random_split`, then creates BehaviorCloning and
runs its training loop. Each observation window keeps its original order.
Each seed has its own `seed_42/`, `seed_43/`, etc., with the collected data.
Each iteration writes weights/history to `trial_0/`, `trial_1/`, etc. inside that seed folder.
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

When using a BehaviorCloning instance directly, its `train()` method returns a
model and its `evaluate(x_eval, y_eval)` method returns MSE without updating weights.
The random internal split may share historical observations between samples;
final testing uses the separate second year, outside the collected teacher horizons.
The trainer uses the supplied split and evaluates after each training epoch.
Hyperparameters are selected manually for now; a future Bayesian search can
supply a new `Hyperparameter` object per trial and compare validation MSE.
`torch.nn.Module.train()` only switches the network's mode; the algorithm's
`train()` method performs the optimization.

## Test a trained policy

`test.py` loads an MLP or LSTM checkpoint and runs it in OpenDSS between the selected
dates. Set `config["architecture"]` manually to `"mlp"` or `"lstm"`, matching the
checkpoint. It reads the architecture dimensions, observation window and device ID
from `training.json` next to the checkpoint. The test main block selects architecture and label manually; set them to match
the training run. Its default selection is MLP without extras. `main()` accepts a single checkpoint. Set the model path, dates, output
folder and seed in its main block, or call:

```python
from test import main

records, summary = main(model_path="results/train/supervised/lstm/seed_42/trial_0/best.pt",
                        config={"case": "data/5bus-switzerland", "architecture": "lstm",
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

| File | Measurements |
| --- | --- |
| `buses.csv` | Voltage in V/pu, angle, and net P/Q injection into connected branches |
| `branches.csv` | P/Q, current in A and angle for each line/transformer terminal and phase |
| `bess.csv` | Requested and measured P/Q, SoC before/after the interval |
| `pv.csv` | Available and measured generation, reactive injection |
| `grid.csv` | Imported/exported power, reactive injection, cost and reward |

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
