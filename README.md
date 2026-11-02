# Local Battery Control with Neural Networks

Imitation learning for local battery energy storage control using PyTorch,
OPF teacher actions, and OpenDSS simulations. Each neural network receives
only measurements from its device's bus and the device's own state, and
outputs commands for that device. The teacher solves the full network OPF
to provide reference actions.

The project includes two editable Git submodules:

- [opf_teacher](https://github.com/TeradaZenichi/opf-teacher): reference actions from optimal power flow.
- [opendss_environment](https://github.com/TeradaZenichi/powerflow-opendss-environment): electrical simulation through a Gymnasium environment.

Research scope, formulation and experiment records are in [docs/](docs/README.md).
The root Python entry points are `train.py` and `test.py`. Essential helpers
live in `tools/`; standalone verification and profile-generation scripts live
in `tools/scripts/`. Network architectures live in `learning/networks/` and
learning algorithms in `learning/algorithms/`; these are currently empty scaffolds.

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
The selected range determines the number of steps, overriding `steps` when an
end date is supplied. Daylight-saving days may contain 92 or 100 intervals.

Without dates, collection defaults to one horizon of data and comparison to
`steps` intervals. Each selected range starts a fresh SoC/history episode.
The saved configuration includes resolved dates, row offset and step count.

## Collect teacher data

```powershell
.\.venv\Scripts\python.exe train.py
```

Edit `config`, `output` and `seed` in the `if __name__ == "__main__"` block,
or call the collector directly:

```python
from train import main

config = {"window": 3, "device": "b1", "horizon": 96, "case": "data/5bus-switzerland",
          "start_date": "2023-01-01", "end_date": "2023-01-02"}
x, y = main(config, output="results/train", seed=42)
```

The teacher advances its own predicted SoC, actions and voltage; no environment
is stepped during collection. The dates select the episode; `horizon` is the
maximum optimization lookahead in intervals. It shrinks near the selected end,
so forecasts never enter the next dataset partition. X contains only the selected bus and battery signals;
Y is that battery's current P/Q action in pu. With window 3, 96 valid steps
produce 94 pairs. Voltage is from the preceding solved interval, initialized
at 1 pu; it never uses the current action's resulting voltage.

The function returns CPU float32 PyTorch tensors: X has shape
`[samples, window, 6]` and Y has shape `[samples, 2]`. The feature order is shared
with `test.py`. For 96 valid intervals and window 3, their shapes are
`[94, 3, 6]` and `[94, 2]`. The tensors are saved as `x.pt` and `y.pt`:

```python
import torch

x = torch.load("results/train/x.pt")
y = torch.load("results/train/y.pt")
```

The original pairs remain in `x.json` and `y.json` for inspection, including
observation timestamps. `config.json` records the seed, settings and
feature/target order. PyTorch is seeded; profiles stay
chronological. Neural-network weight training is not implemented yet.

Supervised algorithm modules can expose `train(x, y, config, output, seed)`, returning
the trained PyTorch module and saving its weights/metrics to `output`.
Once implemented, pass the function directly, for example
`trained_model = main(config, output="results/runs/bc_mlp_w3_seed42_001", seed=42, trainer=behavior_cloning.train)`
after `from learning.algorithms import behavior_cloning`. Without a trainer, `main()` returns X/Y as above.
The algorithm modules are currently empty placeholders; `behavior_cloning.train` is a
proposed entry point, not an implemented function. `torch.nn.Module.train()`
separately switches a network to training mode; the module-level function owns
the optimization loop.

## Compare MPC and a trained NN

`test.py` is the English name of the comparison script. It runs each controller in a separate OpenDSS
context, from the same initial state and exogenous profiles. Both trajectories
use their own executed-state feedback. Set the configuration and trained model
in its main block, or call:

```python
from test import main

records, summary = main(model=trained_model,
                        config={"horizon": 8, "window": 3, "device": "b1",
                                "start_date": "2024-02-29", "end_date": "2024-03-01"},
                        output="results/comparison", seed=42)
```

The CPU `torch.nn.Module` receives float32 input `[batch, window, 6]` and
returns `[batch, 2]` in pu. Input features, in order, are voltage from the
previous interval, current local P/Q demand, current SoC, and previous executed
BESS P/Q. Outputs are charging-positive active power and injection-positive
reactive power. Include any learned preprocessing inside the supplied model.
The first observation is repeated to fill the NN window at episode startup.
This first comparison supports one BESS. Instantiate the trained architecture,
load its weights, and pass the resulting module. With `model=None`, only the
MPC baseline runs.

Collection and comparison both use available PV generation with Q=0; the
teacher optimizes only the BESS. This fixed policy is recorded in the metadata.
Original scenario files are unchanged.

Results include separate `mpc.json` and `nn.json` trajectories and `summary.json`
with executed cost/reward, initial/final SoC, voltage extrema and decision time.
There is no terminal-energy cost adjustment. Generated artifacts stay in the
Git-ignored `results/` folder. No trained-policy performance is claimed yet.

The original detailed teacher/environment check remains available:

```powershell
.\.venv\Scripts\python.exe -m tools.scripts.evaluate_mpc --steps 96 --horizon 8 --label contract
```

## Working on the submodules

Edit either submodule directly and rerun the example. From a fresh clone,
create a work branch inside the submodule before committing changes.
Push submodule commits first, then commit their updated references here.
Use `git push --recurse-submodules=check` when publishing this project.
