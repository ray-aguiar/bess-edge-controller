# Assessing trustworthiness of local BESS imitation control

Research proposal, 2026-11-04. Contributions and experiments below are planned;
the existing integration checks do not establish policy performance or safety
guarantees. This document guides a possible journal extension of an initial
conference study.

## Research question

Under what operating conditions can a BESS controller, observing only its own
bus and battery history, reproduce useful global OPF decisions and maintain
acceptable economic and electrical performance in closed loop?

The study should identify both successful operating conditions and the limits
imposed by missing information. Local imitation of global OPF is established
in the literature; novelty must be demonstrated through a specific method,
analysis or finding about storage dynamics and local information.

## Scope and information available to the controller

The scope is single-phase BESS control, initially using `5bus-switzerland`.
Three-phase control and incentive/budget mechanisms belong to a separate
project. Additional single-phase feeders and battery locations are candidates
for validation of this research question.

| Component | Available information and responsibility |
| --- | --- |
| Teacher | Full network and profiles; optimize once per selected period or optional weekly block to generate reference BESS actions |
| NN | Its bus measurements, its battery state and their local history |
| Local protection, if developed | Local measurements and known device limits |
| Evaluation | Full simulated network; assess impacts without feeding remote measurements to the NN |

The current numerical input is voltage measured before the new action,
current local P/Q demand, pre-action SoC and previous BESS P/Q. A chronological
window forms X; the optimized P/Q action at its last timestamp forms Y. Timestamp and
interval duration are metadata. Prices, forecasts, remote PV and other buses'
measurements are excluded from NN inputs.

Pre-action voltage uses current loads and held previous device powers, with PV
capped by availability. It is recalculated during collection without advancing
SoC and read from the environment's pre-action observation during evaluation.
Previous-interval voltage remains a possible delay-ablation experiment, not
the current input contract. Earlier checkpoints require regeneration/retraining.

Offline examples use teacher-predicted state transitions. Policy evaluation
uses the environment's executed actions and measured feedback. The resulting
change in visited states must be evaluated. PV uses available generation and
zero reactive power in both the student and its reference comparison.

## Candidate contributions

1. Quantify how local state/action history changes the performance gap to the
   global teacher, including situations with similar local histories but
   different reference actions.
2. Develop and evaluate a local protection or uncertainty-handling method,
   with an explicit operating domain and assumptions about unobserved loads
   and generation. This is a candidate method, not an implemented feature.
3. Establish an auditable evaluation of raw NN decisions, device-limit
   corrections and network consequences across unseen operating conditions.

The conference study should answer a complete, focused question. The journal
extension should add a substantive method or analysis and broader evidence.
Changing architectures or adding seeds alone does not establish novelty.

## Trustworthiness in this application

NIST AI RMF 1.0 describes trustworthiness through several context-dependent
characteristics [1]. This study focuses on the engineering dimensions below;
it does not claim comprehensive trustworthiness certification.

| Dimension | Evidence to collect |
| --- | --- |
| Reliability | Episode cost, action errors, operating failures and variation across dates and training seeds |
| Device constraint handling | Raw requested versus executed P/Q, correction frequency/magnitude, SoC and inverter-limit checks |
| Network impact | Voltage/current violation magnitude and duration, losses and location of violations |
| Robustness | Performance under changed load/PV, initial SoC, measurement noise and observation delay |
| Transparency and reproducibility | Feature ablations, saved configurations, model/data provenance and action/measurement traces |

For example, a 40 kW BESS receiving an 80 kW request may execute a limited
command. An admissible executed command is evidence about the combined
controller and limiter. The original NN request must still be assessed.

The current environment enforces modeled BESS power, apparent-power and energy
limits. This does not establish compliance with all network voltage/current
constraints. Remote measurements are available for offline evaluation, but
not for online NN decisions or a proposed local protection mechanism.

Empirical results must specify the tested conditions. Formal guarantees, if
pursued, require explicit assumptions and a separate proof or certificate.
Any proposed fallback requires its own validation; idle operation is a
baseline, not a universal network-safety guarantee. Feature attribution or
uncertainty estimates alone do not certify physical safety.

## Experimental roadmap

| Stage | Question and comparison |
| --- | --- |
| Conference pilot | Can local imitation improve on idle BESS, action persistence and a specified local control rule? |
| Memory study | Compare W=1, 3, 8; ablate previous-action inputs and compare an MLP with a temporal model |
| Information limits | Apply controlled remote load/PV variations and examine local-history similarity versus reference-action variation |
| Closed-loop robustness | Vary initial SoC, seasons, loading and measurement quality; evaluate accumulated deviations |
| Protection study | Compare raw NN requests, current device limiting and any proposed local protection method |
| Generalization | Evaluate held-out periods, BESS locations and additional single-phase feeders |

These are proposed experiments, not completed results. Protection and
uncertainty methods still need selection and implementation. The initial
five-bus system is an integration and pilot case; broader claims require
evidence beyond it.

Reserve 2024 for final testing before constructing windows and teacher horizons.
Split the 2023 pairs randomly into 80% training and 20% internal evaluation;
their windows may share observations, so internal MSE is not an independent-period
generalization estimate. Keep teacher forecasts within the first year, reset
histories at episode boundaries and fit preprocessing on training data only.
Select model settings on internal evaluation data and reserve the final test year.

Start with seeds `[42, 43, 44, 45, 46]`, identical year boundaries and the
same model-selection protocol. Each seed controls the split and training randomness.
Separate variation across these repetitions from
variation across test episodes. Consecutive 15-minute rows are not independent
experimental repetitions; state the aggregation and uncertainty-estimation
procedure used for reported results.

Compare controllers on matching profiles, initial states, device models and
PV policies, with each controller receiving feedback from its own trajectory.
Choose and record the reference MPC horizon; current collection and comparison
defaults differ. A teacher with perfect future profiles is a privileged
information benchmark. Its advantage must be distinguished from NN
approximation error. Use a common terminal-SoC condition or justify an explicit
terminal-energy valuation before comparing costs.

Report P/Q MAE/RMSE in physical units, executed cost and reward, terminal SoC,
battery throughput, corrections, electrical violations and CPU latency.
Choose tolerances and acceptance criteria before the final evaluation.
Report rejected teacher labels and inspect target distributions: the current
flat annual tariff and equal import/export prices may yield little active
dispatch. Keep the documented benchmark and any additional tariff/loading
scenarios clearly identified.

## Current implementation and missing evidence

Available: teacher-only X/Y collection, CPU MLP behavior cloning, shared local
feature encoding, seed control, checkpoint loading, dated environment episodes
and electrical CSV export. The original MPC/NN comparison is preserved in
`tools/scripts/compare_controllers.py`.

`test.py` records requested and executed actions, costs, voltages, branch
currents and battery SoC. `tools/export_results.py` writes temporal CSVs with
one row per timestamp for buses/branches. These support the proposed analysis;
they do not implement uncertainty estimation, a new safety filter or automatic
experimental comparisons.

Still needed for publication: a fixed study protocol, representative training
targets, full held-out evaluation, the selected methodological contribution,
and an updated comparison against the closest literature. No performance,
robustness or global-safety claim follows from the existing smoke tests.

## Related work and publication positioning

- [1] [NIST AI RMF 1.0: AI Risks and Trustworthiness](https://airc.nist.gov/airmf-resources/airmf/3-sec-characteristics/).
  Context-dependent dimensions for defining the evidence assessed here.
- [2] [Karagiannopoulos et al. (2019), Data-Driven Local Control Design](https://doi.org/10.1109/TSG.2019.2905348).
  Local control learned from offline OPF; locality alone is not a new contribution.
- [3] [Dobbe et al. (2020), Toward Distributed Energy Services](https://eta.lbl.gov/publications/toward-distributed-energy-services).
  Reconstruction of centralized OPF from local information; information limits
  must be positioned against this work.
- [4] [Mueller et al. (2025), Comparison of machine learning and MPC methods for control of home battery storage systems](https://doi.org/10.1016/j.apenergy.2025.126465).
  Battery control comparisons including imitation, forecasts and grid constraints.
- [5] [Hou et al. (2024 preprint), Safe Imitation Learning-based Optimal Energy Storage Systems Dispatch](https://arxiv.org/abs/2411.00995).
  Imitation-based storage dispatch with a safety layer; protection alone is not
  an established novelty claim for this project.
- [6] [Yin et al. (2026 preprint), Learning Control Policies from Heterogeneous Multi-Horizon Time Series](https://arxiv.org/abs/2609.22546).
  Temporal representations for battery imitation control; temporal inputs alone
  also require a demonstrated distinction.

This is a starting reading list, not a systematic novelty review. A journal
submission should state the exact methodological and empirical differences
from these works. When extending a conference paper, cite it and identify the
new contributions. Check the target journal's current policy; the
[IEEE PES guidance consulted](https://ieee-pes.org/publications/authors-kit/preparation-and-submission-of-transactions-papers/)
requires at least 60% new content for extensions of PES conference papers.
