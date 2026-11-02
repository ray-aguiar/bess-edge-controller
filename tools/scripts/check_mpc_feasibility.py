import json
from pathlib import Path

import pyomo.environ as pyo

from opf_teacher.teacher import Teacher
from tools.scripts.investigate_mpc import cases


def check_case(case):
    """Construct a feasible zero-VAr dispatch for the single-phase five-bus case."""
    teacher = Teacher(case, formulation="single_phase_socp").build()
    case, model = teacher.case, teacher.model
    dispatch, energy = {}, {}
    for battery in case.bess:
        target = battery.soc_terminal_frac
        if target is None:
            target = battery.soc_init_frac
        change = (target - battery.soc_init_frac) * battery.e_cap_kwh
        duration = case.dt_h * case.n_periods
        dispatch[battery.id] = (max(change, 0.0) / (battery.eta_charge * duration),
                                max(-change, 0.0) * battery.eta_discharge / duration)
        energy[battery.id] = battery.soc_init_frac * battery.e_cap_kwh
    order = [case.root]
    for bus in order:
        order.extend(case.children.get(bus, []))
    impedances = {}
    for bus in order[1:]:
        branch = case.branches[case.parent_branch[bus]]
        r, x = branch.impedance_pu(case.base, case.buses[branch.from_bus])
        impedances[bus] = complex(r, x)
    converged = True
    for time in case.periods:
        demand = {bid: complex(bus.p_load_kw.iloc[time], bus.q_load_kw.iloc[time]) / case.base.s_base_kva
                  for bid, bus in case.buses.items()}
        for battery in case.bess:
            charge, discharge = dispatch[battery.id]
            demand[battery.bus] += (charge - discharge) / case.base.s_base_kva
            model.pch[battery.id, time].set_value(case.base.pu_power(charge))
            model.pdis[battery.id, time].set_value(case.base.pu_power(discharge))
            model.qbess[battery.id, time].set_value(0)
            model.pbess_loss[battery.id, time].set_value(0)
            energy[battery.id] += (battery.eta_charge * charge - discharge / battery.eta_discharge) * case.dt_h
            model.soc[battery.id, time].set_value(case.base.pu_energy(energy[battery.id]))
        for pv in case.pv:
            generation = case.base.pu_power(float(pv.avail_kw.iloc[time]))
            demand[pv.bus] -= generation
            model.ppv[pv.id, time].set_value(generation)
            model.qpv[pv.id, time].set_value(0)
            model.ppv_loss[pv.id, time].set_value(0)
        voltage = {bus: complex(case.grid.v_ref_pu) for bus in order}
        error = float("inf")
        for iteration in range(100):
            current = {bus: (demand[bus] / voltage[bus]).conjugate() for bus in order}
            for bus in reversed(order[1:]):
                parent = case.branches[case.parent_branch[bus]].from_bus
                current[parent] += current[bus]
            updated = {case.root: complex(case.grid.v_ref_pu)}
            for bus in order[1:]:
                parent = case.branches[case.parent_branch[bus]].from_bus
                updated[bus] = updated[parent] - impedances[bus] * current[bus]
            error = max(abs(updated[bus] - voltage[bus]) for bus in order)
            voltage = updated
            if error < 1e-13:
                break
        converged = converged and error < 1e-13
        for bus in order:
            model.v[bus, time].set_value(abs(voltage[bus]) ** 2)
        for bus in order[1:]:
            parent = case.branches[case.parent_branch[bus]].from_bus
            power = voltage[parent] * current[bus].conjugate()
            model.P[bus, time].set_value(power.real)
            model.Q[bus, time].set_value(power.imag)
            model.l[bus, time].set_value(abs(current[bus]) ** 2)
        grid = voltage[case.root] * current[case.root].conjugate()
        if hasattr(model, "pgrid"):
            model.pgrid[time].set_value(grid.real)
        else:
            model.pimp[time].set_value(max(grid.real, 0.0))
            model.pexp[time].set_value(max(-grid.real, 0.0))
        model.qgrid[time].set_value(grid.imag)
    violations = {}
    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        value = pyo.value(constraint.body)
        low = 0.0 if constraint.lower is None else max(pyo.value(constraint.lower) - value, 0.0)
        high = 0.0 if constraint.upper is None else max(value - pyo.value(constraint.upper), 0.0)
        violations[constraint.name] = max(low, high)
    for variable in model.component_data_objects(pyo.Var):
        value = pyo.value(variable)
        low = 0.0 if variable.lb is None else max(variable.lb - value, 0.0)
        high = 0.0 if variable.ub is None else max(value - variable.ub, 0.0)
        violations[variable.name] = max(low, high)
    maximum = max(violations.values())
    sos_ok = all(min(pyo.value(model.pimp[t]), pyo.value(model.pexp[t])) == 0 for t in model.T)
    return {"converged": converged, "max_constraint_or_bound_violation": maximum,
            "worst_constraint": max(violations, key=violations.get), "grid_sos_satisfied": sos_ok,
            "feasible_at_1e_10": converged and sos_ok and maximum <= 1e-10,
            "candidate_objective_chf": pyo.value(model.cost)}


if __name__ == "__main__":
    reports = {name: check_case(case) for name, case in cases().items()}
    output = Path(__file__).resolve().parents[2] / "results/mpc-investigation/feasible-candidates.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
