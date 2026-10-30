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

Requires Python 3.11+, Git LFS, and a valid Gurobi license for the SOCP teacher.

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
fill the history; 96 steps produce 94 pairs. Each call starts a new history.
Use a shorter `HORIZON` for a quick check; each interval requires a new SOCP solve.

## Working on the submodules

Edit either submodule directly and rerun the example. From a fresh clone,
create a work branch inside the submodule before committing changes.
Push submodule commits first, then commit their updated references here.
Use `git push --recurse-submodules=check` when publishing this project.
