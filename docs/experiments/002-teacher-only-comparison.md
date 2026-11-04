# Experiment 002: teacher-only collection and comparison pipeline

Timing note (2026-11-04): the pilot below used previous-interval voltage.
The current pipeline uses current pre-action voltage; regenerate datasets and
retrain checkpoints before comparing results across these input contracts.

Date: 2026-11-01. This is a pipeline validation on the working tree following
commit 949ed67, not a trained-policy performance experiment. The teacher and
environment submodule commits remain those recorded in experiment 001.

## Protocol

The collector uses the teacher's preceding solved SoC and controls to initialize
the next optimization. Local voltage is the preceding solved voltage, initialized
at 1 pu. Current load measurements come from the current profile row. Thus the
voltage feature does not contain the current target action's outcome.
This model-generated dataset differs from the earlier OpenDSS-observation dataset.

Collection and NN/MPC comparison use full available PV generation with Q=0.
Only BESS b1 is controlled. Source scenario files remain unchanged.

The policy input is float32 [batch, window, 6]: previous-interval voltage,
current local P/Q demand, current SoC, and previous executed BESS P/Q.
The output is [batch, 2]: charging-positive P and injection-positive Q in pu.
Preprocessing belongs inside the supplied trained module. NN startup repeats
the first observation until the window fills.

## Checks and outcomes

| Check | Outcome |
|---|---|
| Teacher-only collection, 96 intervals, window 3 | 94 accepted local pairs |
| Environment and comparison-module imports blocked during collection | Collection succeeded |
| Next teacher state compared with previous solved SoC, action and voltage | Causally aligned |
| Legacy detailed MPC contract, 96 intervals, horizon 8 | Same compared costs, BESS actions and SoC as experiment 001 |
| Comparison environments | Distinct OpenDSS contexts and independent measured-state feedback |
| Full-day comparison, horizon 8 | 96 MPC and 96 NN-fixture steps completed |
| PV commands in both trajectories | Available generation and Q=0 within numerical tolerance |
| Executed reward and SoC | Reward equalled negative executed cost; SoC stayed within bounds |
| CPU model instantiated and loaded from saved weights | Accepted by the comparison runner |
| NN inference with teacher construction blocked | Succeeded |

The NN fixture was a constant-output module requesting 5 kW charging and
4 kvar injection, not a trained controller. It reached full SoC and exercised
environment action adjustment. Its cost is not evidence of learning quality;
there is no terminal-energy adjustment in the comparison summary.

Local evidence is under results/teacher-only-validation/ and
results/comparison-validation/, including input/output traces, saved fixture
weights, summaries and runtime source hashes. These artifacts are ignored by Git.
The next study must train a policy and evaluate unseen periods.

## Date-range selection checks

The comparison runner is now in `tools/scripts/compare_controllers.py`;
`test.py` loads a trained MLP for policy-only evaluation. A later check selected
2024-02-29 12:00 to 13:00 in Europe/Zurich: four intervals, two local pairs
with window 3, and maximum optimization horizon 2. Both controller environments
matched the selected timestamps, demand P/Q, PV availability and 2024 tariff.
No forecast crossed the exclusive end date. Separate selection checks covered
the leap day, 23-hour and 25-hour Swiss daylight-saving days, explicit UTC
inputs and naive-timestamp datasets. All 47 environment tests passed,
including two tests for the new backward-compatible start_step offset.

## PyTorch tensor checks

The collector now returns CPU float32 X/Y tensors and saves x.pt/y.pt alongside
the original JSON pairs. Six intervals on 2024-02-29, from 12:00 to 13:30 in
Europe/Zurich, produced shapes [4, 3, 6] and [4, 2] with window 3 and horizon 2.
A single interval with window 1 preserved shapes [1, 1, 6] and [1, 2].
Checks confirmed chronological windows, the shared evaluation feature order,
values matching the JSON pairs and exact save/load round trips. A small PyTorch
linear model accepted X, produced Y-shaped outputs and computed finite MSE
gradients. This was a tensor interface check, not trained-policy evaluation.
Artifacts are under the Git-ignored results/tensor-validation/ directory.
