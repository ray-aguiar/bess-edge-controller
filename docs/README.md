# Documentation

This repository develops and evaluates a local BESS controller for a research
publication. The policy controls one battery using its bus measurements, battery
state and history. Central orchestration and controllers for PV/EV charging are
separate research extensions; the reusable submodules retain their broader APIs.

- [Usage](usage.md): configuration, data collection, architectures, features, outputs and diagnostics.
- [Formulation](formulation.md): teacher objective, local observations/actions and timing.
- [Experiments](experiments.md): evaluation protocol, comparisons and metrics.
- [Research direction](journal-research.md): local BESS research question and publication roadmap.
- [Experiment 001](experiments/001-teacher-environment.md): teacher/environment integration checks.
- [Experiment 002](experiments/002-teacher-only-comparison.md): teacher-only collection pilot.
- [Experiment 003](experiments/003-mse-reward-selection.md): historical reward-selection pilot; current training selects by MSE.

Implemented policies are MLP and LSTM with supervised behavior cloning; other
learning modules are scaffolds. Generated datasets, models and simulation outputs
belong in the Git-ignored `results/` directory. Document evidence and conclusions
here so that publication records do not depend on locally saved artifacts.
