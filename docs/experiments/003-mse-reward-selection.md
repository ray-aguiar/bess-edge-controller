# Experiment 003: MSE versus reward checkpoint selection

Date: 2026-11-06. This is a single-seed pilot using the current working tree,
not a final statistical or full-year policy evaluation.

## Protocol

- Reused `results/train/seed_43/x.pt` and `y.pt`: 5,568 samples from
  2023-01-01 to 2023-02-28, end exclusive, in Europe/Zurich.
- Local BESS b1, six features, window 96, repeat-first startup padding,
  one full-period teacher optimization and fixed available PV with Q=0.
- Same 80/20 random MSE split, seed 43, CPU MLP with 64 hidden units,
  Adam learning rate 1e-3, batch 64 and 100 epochs as the saved original run.
- Reward validation: 2023-03-01 to 2023-03-10, end exclusive, 864 intervals.
  The environment resets to 50% SoC with seed 43 before each epoch's rollout.
- Test comparison after selection: 2024-01-01 to 2024-01-10, end exclusive,
  864 intervals, environment seed 42, initial SoC 50% for both checkpoints.
- Training still minimizes MSE. Independent checkpoints select minimum MSE
  and maximum raw cumulative reward; reward remains negative executed cost.
  There is no terminal-SoC requirement or terminal-energy adjustment in evaluation.

The new `tools/evaluation.py` shares the same physical rollout between
per-epoch reward validation and `test.py`. It avoids retaining episode records
when called during training. The default entry point rejects overlapping
teacher-training/reward-validation dates before generating teacher labels.

Repository bases were main `5c88c847`, teacher `1a1a8184`, and environment
`63cc426b`, with uncommitted changes. Completion-time source, tensor and
checkpoint hashes are saved under `results/reward_selection/fingerprints.json`.
The original user checkpoints and test artifacts were preserved.

## Results

| Metric | Minimum-MSE checkpoint | Maximum-reward checkpoint |
|---|---:|---:|
| Selected epoch | 89 | 70 |
| Internal validation MSE, pu squared | 1.09359e-8 | 1.20436e-6 |
| March validation reward | -1529.557858 | -1502.849226 |
| March validation final SoC | 99.0197% | 10.0000% |
| January 2024 test reward | -3847.589009 | -3818.860614 |
| January 2024 final SoC | 90.1770% | 10.0000% |
| Test intervals with P adjustment above 1e-4 kW | 41 | 728 |
| Test charge + discharge energy, kWh | 100.7493 | 37.4579 |

The reward-selected checkpoint improves the measured test reward by
28.728395, despite much worse imitation MSE. The minimum-MSE weights are
bitwise identical to the previous seed-43 checkpoint: adding reward evaluation
did not change the reproduced MSE training outcome.

Training with 100 reward evaluations took 338.7 seconds, excluding teacher
data generation and the final two test rollouts. A cached nine-day evaluation
took approximately 2.9 seconds on this machine.

## Interpretation and limits

The two criteria select different policies in this pilot. Improvement is real
for the specified raw reward, but the final battery energy differs by 80.177 kWh.
The reward-selected policy ends at the lower SoC bound and requires more
environment action adjustments. It must not be interpreted as having learned
constraint handling or cyclic battery operation independently of the environment.

This supports keeping both selection criteria and logging final SoC and action
adjustments. It does not establish an economic ranking at matched terminal
energy, robustness across seeds, or seasonal generalization. The 2024 dates
were already examined in earlier pilot analysis; this is not a blind final test.

## Outputs and reproduction

- `results/reward_selection/seed_43/trial_0/`: weights and per-epoch history.
- `best_mse.pt` and `best_reward.pt`: independent selected models.
- `best.pt`: compatibility copy of the minimum-MSE selection.
- `test/best_mse/` and `test/best_reward/`: separate temporal test exports.
- `results/reward_selection/comparison.json`: complete comparison metrics.

The current `train.py` main block runs both selections using its `reward_eval`
dates. `test.py` evaluates both new checkpoints into separate directories.
This recorded pilot reused saved tensors instead of regenerating the dataset.
