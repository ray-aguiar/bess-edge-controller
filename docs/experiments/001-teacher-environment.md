# Experiment 001: teacher/environment compatibility

Recorded on 2026-11-01 from completed local runs; no new simulations were
performed for this documentation.

## Configuration

- Main repository: 949ed67.
- Teacher: 40eb2d51506dab22f97450ac0bd4c2ce899d7a31.
- Environment: bfe3b4723289e299d7a74e9e2351a807f065e709.
- Case: data/5bus-switzerland, original 1,000-kVA and 11-kV bases.
- Episode: 96 intervals of 15 minutes, starting at 2023-01-01 00:00 Europe/Zurich.
- BESS b1: bus bus_004, 100 kWh, 40-kW charge/discharge limits, 50 kVA,
  initial and terminal-target SoC 50%.
- PV pv1: bus bus_005, 260-kW rating, 300-kVA inverter.
- Solver: Clarabel 0.11.1; Pyomo 6.10.1; NumPy 2.5.4; SciPy 1.18.1.
- Original tariff: buy price equals sell price; no forecast-error perturbations.

Input SHA-256 hashes (file bytes at documentation time):

| File under data/5bus-switzerland/ | SHA-256 |
|---|---|
| config.json | da57f1870c23722e33ba8e29f6c055411691533dc36bb699c96cb81f8ca3347e |
| devices.json | 9b10a2980c367540ec16428f44b10355462dbd2fbf646e951bfae3cd0d633474 |
| demand.csv | c6e6a9285bbeb411a9739980ebee821c43846eed578f4bb6d47a7905591f5a84 |
| price.csv | 4843a4858b5d54b889051234682d59260c7ac7af8964dfce8e8e3185e19da615 |
| pv.csv | f5bbb4e4fe64f84731d91ac2a089b37e95c70edfd1dfdef9dffd4c924b24710d |

## Reproduction

These commands describe the recorded pipeline at commit 949ed67. The detailed
contract evaluation is now available as `python -m tools.scripts.evaluate_mpc`; current train.py
uses teacher-only states and a fixed PV policy by default, so its new dataset
is not identical to the historical environment-observation dataset.

~~~powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_mpc --steps 96 --horizon 8 --label fixed
.\.venv\Scripts\python.exe -m scripts.evaluate_mpc --steps 96 --horizon 24 --label fixed
.\.venv\Scripts\python.exe train.py
~~~

## Recorded results

| Metric | Horizon 8 / 2 h | Horizon 24 / 6 h |
|---|---:|---:|
| Optimal / tight / valid-label steps | 96 / 96 / 96 | 96 / 96 / 96 |
| Executed cost, CHF | 144.557716801 | 144.557670593 |
| Cumulative reward | -144.557716801 | -144.557670593 |
| Maximum interval cost error, CHF | 6.88e-8 | 6.87e-8 |
| Maximum SoC prediction error, fraction | 5.92e-11 | 4.97e-11 |
| Timestamp / SoC-feedback mismatches | 0 / 0 | 0 / 0 |

The pair-collection run produced 94 local pairs with window 3. After the
refactor, all pairs were identical and both fixed-horizon trajectories
retained identical compared costs, BESS actions and executed SoC.
The associated suites passed 101 teacher and 45 environment tests.

A separate temporary scenario with sell/buy ratio 0.2 and horizon 24 also
completed 96 optimal, tight, valid steps. Executed cost was 205.591243032 CHF,
versus 223.493220645 CHF for idle BESS. Final SoC was 51.0013% versus 50%.
This was a hypothetical tariff check; it did not alter the original data,
has no terminal-energy cost adjustment, and is not a measured Swiss export contract.

## Interpretation and evidence

These runs validate integration on one day. They do not establish performance
over the full dataset, superiority of one solver in general, or student-policy
performance. Original symmetric tariffs provided very weak active-power
dispatch, so broader data coverage is needed before neural-network training.
The inherited electrical equivalent is configured at 60 Hz in Master.dss;
the Swiss profiles do not establish that it represents an actual Swiss feeder.

Local artifacts are under results/mpc-96-h8-fixed/,
results/mpc-96-h24-fixed/, results/mpc-96-h24-tariff20-fixed/,
results/mpc-fixes/ and results/refactor-validation/; Git ignores them.
Configuration and runtime hashes were saved in the local validation JSON files.
The discounted-export run is recorded here but has no dedicated reproduction CLI.
