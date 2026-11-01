# Local Battery Control with Neural Networks

Imitation learning for local battery energy storage control using PyTorch,
OPF teacher actions, and OpenDSS simulations. Each neural network receives
only measurements from its device's bus and the device's own state, and
outputs commands for that device. The teacher solves the full network OPF
to provide reference actions.

The project includes two editable Git submodules:

- [opf_teacher](https://github.com/TeradaZenichi/opf-teacher): reference actions from optimal power flow.
- [opendss_environment](https://github.com/TeradaZenichi/powerflow-opendss-environment): electrical simulation through a Gymnasium environment.

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

## Run the example

```powershell
.\.venv\Scripts\python.exe train.py
```

The example collects local observation/action pairs in pu for BESS `b1` at
`bus_004` over one day of the Swiss five-bus scenario. Each step solves the
remaining SOCP horizon and executes the full teacher action in OpenDSS.
Set `DEVICE_ID` to select the BESS and `WINDOW` to set the number of observed
states in each X. With `WINDOW = 3`, X contains `[state(t-2), state(t-1), state(t)]`
and Y contains only the selected device's action at `t`. The first two steps
fill the history; 96 valid steps produce 94 pairs. Labels must pass the teacher's
physical-consistency and SOCP checks. Each call starts a new history.
Training consumes the shared `mpc.rollout()` one interval at a time, using the
remaining episode as its forecast horizon. `mpc.py` uses a fixed rolling horizon.
Use a shorter `HORIZON` for a quick check; each interval requires a new SOCP solve.

## Run MPC

```powershell
.\.venv\Scripts\python.exe mpc.py
```

`mpc.py` starts with 96 simulation steps (one day) and an eight-interval SOCP forecast
horizon (two hours). Each step updates the teacher with the current OpenDSS
observation, solves the next `HORIZON` intervals, and applies only the first
action for all devices. The horizon then advances by one interval. Set `STEPS`
and `HORIZON` at the top of the script. `main()` returns the states, requested
and executed actions, predicted battery SoC, costs, voltages, solver summaries,
and next observations for inspection. The objective covers the entire forecast
horizon; compare its first-interval cost with the environment cost and reward.

Compare a complete day with an idle-battery baseline:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_mpc --steps 96 --horizon 8 --label fixed
.\.venv\Scripts\python.exe -m scripts.evaluate_mpc --steps 96 --horizon 24 --label fixed
```

The evaluation saves per-step comparisons, raw records, and summaries under
`results/`, which is ignored by Git. Both horizons completed 96 optimal steps
with valid labels on the original 1,000-kVA base.

## Working on the submodules

Edit either submodule directly and rerun the example. From a fresh clone,
create a work branch inside the submodule before committing changes.
Push submodule commits first, then commit their updated references here.
Use `git push --recurse-submodules=check` when publishing this project.
