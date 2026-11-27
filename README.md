# Local BESS Control

Train and evaluate PyTorch policies that control one BESS using measurements
from its own bus and battery history. A global OPF teacher provides reference
actions; OpenDSS evaluates closed-loop operation.

## Install

Requires Python 3.11+, Git and Git LFS.

```powershell
git lfs install
git clone --recurse-submodules https://github.com/ray-aguiar/bess-edge-controller.git
cd bess-edge-controller
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an existing checkout, run `git submodule update --init --recursive`.

## Generate data and train

Set the case, dates, window and `extra_features` in [train.py](train.py), then
run:

```powershell
.\.venv\Scripts\python.exe train.py
```

Training reuses matching `x.pt`/`y.pt` tensors when available, otherwise it
generates the teacher dataset first. The current defaults use an MLP and the
normalized grid buy price as an extra input; set `"extra_features": {}` to
disable optional inputs. Datasets are saved under `results/datasets/`, and
changing the architecture lets you train another model from the same data.

The best validation-MSE model is saved as `best.pt` under
`results/train/supervised/<architecture>/<dataset_name>/seed_42/trial_0/`.
`dataset_name` identifies the selected case, dates, window and features.

## Test

In [test.py](test.py), select the model architecture, matching
`MODEL_EXTRA_FEATURES`, seed and test dates. For the training defaults, set
`MODEL_EXTRA_FEATURES = {"grid": ["buy_price_01"]}`.

```powershell
.\.venv\Scripts\python.exe test.py
```

Results are saved under `results/test/`, including cost/reward, SoC and electrical
CSV measurements. Date ranges include `start_date` and exclude `end_date`.

For comparison baselines, run [test_no_bess.py](test_no_bess.py) to disable
the battery or [test_rule_based.py](test_rule_based.py) to evaluate the
`price_moving_average` dispatch rule.

See [docs/](docs/README.md) for configuration details, formulation and research experiments.
