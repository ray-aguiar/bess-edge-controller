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
generates the teacher dataset first. Set `"extra_features": {}` to disable
optional inputs. Datasets are saved under `results/datasets/`; changing the
architecture lets you train another model from the same data.

The best validation-MSE model is saved as `best.pt`. Current defaults produce:

```text
results/train/supervised/lstm/extra_features/seed_42/trial_0/best.pt
```

With `extra_features={}`, the `extra_features` folder is omitted.

## Test

In [test.py](test.py), select the matching model, seed and test dates.
For the current training defaults, set:

```python
architecture = "lstm"
label = "extra_features"
```

```powershell
.\.venv\Scripts\python.exe test.py
```

Results are saved under `results/test/`, including cost/reward, SoC and electrical
CSV measurements. Date ranges include `start_date` and exclude `end_date`.

See [docs/](docs/README.md) for configuration details, formulation and research experiments.
