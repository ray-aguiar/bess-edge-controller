# Local imitation-control formulation

This describes the implemented single-phase case and local student.
The [journal research direction](journal-research.md) defines the proposed
trustworthiness study and its evidence requirements.
Source: [policy evaluation](../test.py),
[comparison](../tools/scripts/compare_controllers.py), [pair collection](../train.py) and the
[teacher model](../opf_teacher/opf_teacher/opf/single_phase/distflow_socp.py).

## Timing, units and signs

Each interval lasts $\Delta t=0.25$ h. During offline collection, the teacher
advances its predicted SoC and previous controls without executing OpenDSS.
During evaluation, each controller executes its action in its own environment;
measured SoC and executed previous controls feed its next decision.

Input profiles use kW and CHF/kWh. The observation/action API uses pu on the
total-network base: 1,000 kVA and 11 kV. SoC is a fraction.
Battery active power is positive for charging and negative for discharging;
positive device reactive power means injection. Positive grid power here means import.

## Global teacher

At time $t$, minimize energy cost over $K$ forecast intervals:

$$
\min J_t=\sum_{k=t}^{t+K-1}\Delta t
\left(c_k^{buy}P_k^{import}-c_k^{sell}P_k^{export}\right).
$$

Here powers are in kW, $c^{sell}=\rho c^{buy}$ and $J_t$ is in CHF.
The original case has $\rho=1$. Network and device losses affect cost through
the constraints; the implemented objective has no separate degradation penalty.
Future loads and PV come directly from the scenario profiles, so this is a
perfect-profile benchmark rather than a forecast-error experiment.

For a radial branch $i\rightarrow j$, the following equations use pu.
$P,Q$ are sending-end flows, $v=|V|^2$, and $\ell=|I|^2$.
With net device/load injections $p_j^{inj},q_j^{inj}$:

$$
P_{ij}-r_{ij}\ell_{ij}+p_j^{inj}=\sum_{j\rightarrow m}P_{jm},
\qquad
Q_{ij}-x_{ij}\ell_{ij}+q_j^{inj}=\sum_{j\rightarrow m}Q_{jm}.
$$

$$
v_j=a_{ij}^2\left[v_i-2(r_{ij}P_{ij}+x_{ij}Q_{ij})
+(r_{ij}^2+x_{ij}^2)\ell_{ij}\right],
\qquad P_{ij}^2+Q_{ij}^2\leq v_i\ell_{ij}.
$$

$a_{ij}$ is the branch tap ratio. Configured voltage, current and grid-exchange
bounds complete the network model. The root bus has fixed voltage and grid injection.
The inequality is the SOCP relaxation; its tightness must be checked after solving.

For each battery, the following equations use kW, kvar and kWh.
Let $e_k$ be stored energy before interval $k$:

$$
P_k^B=P_k^{ch}-P_k^{dis},\qquad
e_{k+1}=e_k+\Delta t\left(\eta_{ch}P_k^{ch}
-P_k^{dis}/\eta_{dis}-L_k^B\right).
$$

$$
(P_k^B)^2+(Q_k^B)^2\leq S_B^2,\qquad
L_k^B\geq L_{rated}^B(Q_k^B/S_B)^2.
$$

Charge/discharge powers and stored energy have configured bounds. Battery
inverter loss is debited from stored energy. The model allows nonnegative
charge and discharge variables simultaneously; diagnostics check their physical
consequences. The current cyclic terminal target is 50% SoC at each horizon end.
Initial SoC follows the teacher's predicted state during offline collection
and the environment's executed state during MPC evaluation.

The teacher also supports curtailable PV, with daytime constraints including
$0\leq P^{PV}$, $P^{PV}+L^{PV}\leq P^{available}$,
$(P^{PV})^2+(Q^{PV})^2\leq S_{PV}^2$ and
$L^{PV}\geq L_{rated}^{PV}(Q^{PV}/S_{PV})^2$.
Night reactive control is disabled in this case.

Continuous cases default to Clarabel; SOS cases retain Gurobi. Internal
scaling does not change public units. Teacher quality checks cover scaled
feasibility, physical device consistency and branch-cone tightness.

## Local student and supervised pairs

For BESS $d$, the current selector retains these pre-action signals:

| Source | Fields |
|---|---|
| Its bus | v_before_pu, p_load_pu, q_load_pu |
| Its battery | soc_before_frac, previous_p_pu, previous_q_pu |
| Metadata | timestamp, dt_h |

Other buses, grid prices, forecasts and remote PV measurements are excluded.
The voltage field is measured before the new action, with current loads and
previous device powers held; PV is capped by current availability. Collection
uses a radial fixed-power snapshot without advancing energy; evaluation uses
the environment's pre-action observation. The initial snapshot uses zero
previous device powers. This keeps it independent of the current target action. The shared feature
order is voltage, local P demand, local Q demand, SoC, previous BESS P and Q.
Timestamp and interval duration remain metadata, not numeric NN features.
Data and model configurations record `voltage_source="pre_action"`; models
trained with previous-interval voltage must be retrained before evaluation.

$$
X_t^d=[o_{t-W+1}^d,\ldots,o_t^d],\qquad
Y_t^d=[P_t^{B,*},Q_t^{B,*}],\qquad
\widehat{Y}_t^d=\pi_\theta(X_t^d).
$$

$Y_t^d$ selects the device's first action from the global OPF reference. It is a
requested action; executed actions and their adjustments are logged separately.
Both target powers are in pu in the pair contract; multiply by the network
power base to report kW and kvar.
With $W=3$, the first two steps fill the history and 96 valid steps yield 94 pairs.
History resets between episodes. Labels require quality["valid_for_imitation"].

BehaviorCloning uses mean squared error between predicted and teacher actions
and trains the MLP with Adam. Hyperparameters are supplied manually; validation
uses separately prepared date ranges. Any future learned preprocessing must
be fitted on the training partition only. Since local inputs omit information
used by the teacher, exact imitation is not guaranteed.

## Closed-loop evaluation

Execute the student command in OpenDSS and use the resulting measured state
for the next input. Compare executed interval cost with the teacher's
first-interval predicted cost, and use $reward_t=-cost_t^{executed}$.
The horizon objective $J_t$ covers several intervals.

The MPC comparison helper defaults to $K=8$ and train.py to $K=96$.
test.py loads a trained policy and does not solve an MPC. Dates select a
half-open dataset range; K is the maximum teacher/MPC forecast lookahead.
The final horizons shrink to stay inside the selected range, avoiding forecast
leakage across partitions. Choose a consistent K before a learning study. Both use fixed, uncurtailed PV with Q=0, so the two
controllers have the same control authority over the BESS.

The supplied CPU model accepts float32 tensors of shape [batch, window, 6]
and returns [batch, 2] powers in pu. Any learned preprocessing belongs inside
the model. During startup, evaluation repeats the first observation until the
window fills. Include this startup convention in the learning experiment;
current pair collection retains its two-step warm-up when the window is three.
