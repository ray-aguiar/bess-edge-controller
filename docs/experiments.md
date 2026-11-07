# Experiment plan

The main question is how closely a battery controller with local measurements
can reproduce the decisions and closed-loop performance of a global teacher.

The [journal research direction](journal-research.md) extends this question
to reliability, local information limits, constraint handling and robustness.
It distinguishes implemented diagnostics from proposed methods and evidence.

## Sequence

| ID | Question | Status |
|---|---|---|
| 001 | Do teacher predictions, executed actions, SoC and costs agree? | Completed for the documented one-day scenarios |
| 002 | Can we collect valid local pairs and compare independent rollouts? | Pipeline pilot completed; broader data coverage planned |
| 003 | How do MSE and reward checkpoint selection compare for a small CPU MLP? | Historical single-seed pilot; reward selection removed |
| 004 | Does observation-window length improve imitation? | Planned |
| 005 | How does the student behave when its own actions change future observations? | Planned |
| 006 | How sensitive are results to season, initial SoC and forecast errors? | Planned |

See [experiment 001](experiments/001-teacher-environment.md) for the current
baseline and [experiment 002](experiments/002-teacher-only-comparison.md) for
the teacher-only collection and comparison pilot. Integration success is not
evidence of trained-policy performance.

See [experiment 003](experiments/003-mse-reward-selection.md) for the initial
trained-MLP comparison of MSE and closed-loop reward checkpoint selection.

## Dataset protocol

Before an experiment, fix the teacher optimization period/block length, episode/reset rules,
terminal SoC target, numeric feature order and treatment of timestamps.
Preserve local-only inputs and pre-action timing.

Split chronological blocks before constructing windows and solving label
horizons. Keep each window and forecast horizon within its partition, and
reset history at episode boundaries. Fit preprocessing on training data only;
keep validation for model selection and test data untouched until final evaluation.
Record the selected dates rather than assuming all of 2023/2024 was used.

The current training entry point uses a random internal MSE split and saves
`best.pt` after validation at each epoch. No reward is used for model selection.
The test entry point measures closed-loop performance on the separate test period,
including cost, reward and final SoC. Experiment 003 records the earlier reward-selection pilot.

Use the existing Swiss profiles and document their origin in
[SOURCES.md](../data/5bus-switzerland/SOURCES.md). Count attempted, rejected and
accepted labels, report rejection reasons, and inspect active/reactive target
distributions. The current tariff is flat within each year, with sale price equal to 20% of
purchase price. Earlier symmetric-tariff runs produced little active battery
dispatch. Keep these scenarios separate and inspect the new target distribution.

## Comparisons and metrics

Start with the global teacher, an idle BESS, a previous-action predictor and
a small MLP. For the window study, compare $W=1,3,8$ under the same split and
training budget; these are proposed settings.

Report active/reactive MAE and RMSE in physical units, and runtime on the same
CPU. In closed loop, also report executed cost and reward, minimum/maximum
voltage, SoC violations, requested/executed action differences, losses, rejected
commands and battery energy throughput. Report final SoC so that cost savings
are not confused with consuming additional stored energy.

The evaluator uses the same fixed PV generation and zero reactive
PV power for both controllers. It controls one selected BESS, keeps other BESS commands at zero, uses environment
action-limit handling and repeats the first local observation to fill the NN
window at startup. Report requested/executed adjustments and the effects of
startup padding; a numerical smoke test is not trained-policy validation.

## Experiment records

Create one concise document per experiment under docs/experiments/.
Record the question, date, repository/submodule commits, data/configuration
hashes, time range, initial/terminal SoC, horizon, window, solver or model
settings, seed, commands, metrics, conclusions and limitations.

Store generated outputs under a distinct results/ directory and record
their relative paths. Commit the settings and a concise result table in
docs/, so ignored artifacts are not the only record of an experiment.
