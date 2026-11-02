# Research documentation

This repository studies whether a small neural network can imitate a global
OPF controller using only measurements available at one battery's bus.

## Scope

| Component | Responsibility |
|---|---|
| This repository | Scenarios, data collection, PyTorch policies, experiments |
| opf_teacher | Global optimization and solution-quality diagnostics |
| opendss_environment | Execute commands and measure the next electrical state |

Only train.py and test.py are Python entry points at the repository root.
Essential data modules are in tools/; other scripts are in tools/scripts/.
Run diagnostics from the root with `python -m tools.scripts.evaluate_mpc`.

The initial controller operates BESS b1 at bus_004 in data/5bus-switzerland.
The teacher observes the whole network; the student receives only its bus
measurements, battery state and local history.

## Documents

- [Formulation](formulation.md): teacher, local policy, observations and targets.
- [Experiment plan](experiments.md): questions, comparisons and evaluation rules.
- [Experiment 001](experiments/001-teacher-environment.md): completed integration checks.
- [Experiment 002](experiments/002-teacher-only-comparison.md): teacher-only data and independent comparison checks.

## Current state

train.py collects local X/Y pairs from a teacher-only trajectory and returns
CPU float32 tensors [samples, window, 6] and [samples, 2]. It saves x.pt/y.pt,
the original JSON pairs and configuration/output/seed metadata.
The teacher's predicted SoC and previous
controls initialize the next step; no environment is stepped during collection.
Voltage features come from the previous interval, initialized at 1 pu.

The renamed test.py accepts a trained CPU PyTorch model and compares it with
MPC in independent OpenDSS environments. Its default PV policy is identical
fixed generation with zero reactive power in both trajectories. Model=None
runs only the baseline. Neural-network weight training and model export remain
to be implemented; learning/networks/ and learning/algorithms/ are empty scaffolds.

Both scripts accept start_date/end_date. The interval includes the start and
excludes the end; naive dates use Swiss local time for the current dataset.
The environment receives the corresponding dataset row offset.

The next milestone is to choose the dataset horizon and partitions, train a
small MLP on CPU, and evaluate it on unseen episodes. The saved feature order
must match the model, including preprocessing and startup padding.

Keep formulations, experiment settings and concise conclusions in docs/.
Generated trajectories, plots, datasets and model files go under results/,
which Git ignores. A documented experiment must remain understandable without
access to those local artifacts.
