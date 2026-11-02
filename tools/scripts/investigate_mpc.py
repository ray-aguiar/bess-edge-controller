import json
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from opf_teacher.normalization import PUNormalizer
from opf_teacher.opf.single_phase import BessState, BusState, PvState, load_case
from opf_teacher.opf.single_phase.results import attach_results
from opf_teacher.teacher import Teacher


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "mpc-investigation"


def cases():
    source = load_case(ROOT / "data" / "5bus-switzerland")
    records = json.loads((ROOT / "results/mpc-64-h8/records.json").read_text(encoding="utf-8"))
    selected = {}
    for step in (0, 38, 46):
        teacher = Teacher(source.horizon(step, periods=8), formulation="single_phase_socp")
        teacher.observe_dict(records[step]["info"]["observation_before"])
        selected[f"step{step}"] = teacher.case
    for step in (64, 68):
        fixture = json.loads((OUTPUT / f"observed-step{step}.json").read_text(encoding="utf-8"))
        teacher = Teacher(source.horizon(step, periods=fixture["periods"]), formulation="single_phase_socp")
        values = fixture["states"]
        teacher.observe(buses={int(bid): BusState(**state) for bid, state in values["buses"].items()},
                        bess={did: BessState(**state) for did, state in values["bess"].items()},
                        pv={did: PvState(**state) for did, state in values["pv"].items()})
        selected[f"step{step}"] = teacher.case
    return selected


def build_variant(case, variant):
    teacher = Teacher(case, formulation="single_phase_socp")
    if variant.startswith("base100"):
        teacher.case.base.s_base_kva = 100.0
        teacher.normalizer = PUNormalizer(teacher.case.base.s_base_kva, teacher.case.base.v_base_kv)
    teacher.build()
    model = teacher.model
    if "loss" in variant and "unit_loss" not in variant:
        for devices, constraints in ((teacher.case.bess, model.bess_q_loss), (teacher.case.pv, model.pv_q_loss)):
            for device in devices:
                scale = teacher.case.base.pu_power(device.q_loss_rated_kw)
                if variant == "loss_power":
                    scale = teacher.case.base.pu_power(device.s_max_kva) ** 2
                if scale > 0:
                    for time in model.T:
                        constraint = constraints[device.id, time]
                        constraint.set_value(constraint.body / scale <= 0)
    if "unit_loss" in variant:
        normalize_losses(teacher)
    if "continuous" in variant and hasattr(model, "no_roundtrip"):
        model.no_roundtrip.deactivate()
    if "net_grid" in variant and not hasattr(model, "pgrid"):
        if teacher.case.grid.feed_in_ratio != 1.0:
            raise ValueError("The net-grid experiment requires equal buy and sell tariffs")
        grid = teacher.case.grid
        base = teacher.case.base
        model.pgrid = pyo.Var(model.T, bounds=(-base.pu_power(grid.p_export_max_kw), base.pu_power(grid.p_import_max_kw)))
        model.no_roundtrip.deactivate()
        for time in model.T:
            balance = model.balance_p[teacher.case.root, time]
            balance.set_value(balance.body - model.pimp[time] + model.pexp[time] + model.pgrid[time] == 0)
            model.pimp[time].fix(0)
            model.pexp[time].fix(0)
        model.cost.set_value(base.s_base_kva * teacher.case.dt_h * sum(float(teacher.case.price.iloc[t]) * model.pgrid[t] for t in model.T))
    return teacher


def normalize_losses(teacher):
    model = teacher.model
    model.bq = pyo.Var(model.S, model.T, bounds=(-1, 1))
    model.bl = pyo.Var(model.S, model.T, bounds=(0, 1))
    model.gq = pyo.Var(model.G, model.T, bounds=(-1, 1))
    model.gl = pyo.Var(model.G, model.T, bounds=(0, 1))
    model.normalized_losses = pyo.ConstraintList()
    for devices, reactive, loss, q_normalized, loss_normalized, original in (
        (teacher.case.bess, model.qbess, model.pbess_loss, model.bq, model.bl, model.bess_q_loss),
        (teacher.case.pv, model.qpv, model.ppv_loss, model.gq, model.gl, model.pv_q_loss),
    ):
        original.deactivate()
        for device in devices:
            rating = teacher.case.base.pu_power(device.s_max_kva)
            rated_loss = teacher.case.base.pu_power(device.q_loss_rated_kw)
            for time in model.T:
                key = device.id, time
                model.normalized_losses.add(reactive[key] == rating * q_normalized[key])
                model.normalized_losses.add(loss[key] == rated_loss * loss_normalized[key])
                if rated_loss > 0:
                    model.normalized_losses.add(q_normalized[key] ** 2 <= loss_normalized[key])


def solve_variant(case, variant):
    teacher = build_variant(case, variant)
    solver = pyo.SolverFactory("gurobi_direct")
    solver.options["TimeLimit"] = 20
    if "strict" in variant:
        solver.options.update({"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9,
                               "BarQCPConvTol": 1e-10, "MIPGap": 1e-8, "MIPGapAbs": 1e-8})
    result = solver.solve(teacher.model, load_solutions=False)
    native = solver._solver_model
    row = {"variant": variant, "termination": str(result.solver.termination_condition),
           "native_status": native.Status, "solutions": native.SolCount, "runtime_s": native.Runtime, "is_mip": bool(native.IsMIP)}
    if str(result.solver.termination_condition) != "optimal":
        return row
    teacher.model.solutions.load_from(result)
    if "net_grid" in variant and teacher.model.pimp.ctype is pyo.Var:
        for time in teacher.model.T:
            power = pyo.value(teacher.model.pgrid[time])
            teacher.model.pimp[time].unfix()
            teacher.model.pexp[time].unfix()
            teacher.model.pimp[time].set_value(max(power, 0.0))
            teacher.model.pexp[time].set_value(max(-power, 0.0))
    solved = attach_results(teacher.model, teacher.case, "optimal")
    row.update({"objective_chf": solved.summary.objective_cost, "tightness": solved.summary.socp_tightness,
                "cone_gap_pu2": solved.summary.socp_gap_max_pu2, "cone_current_error_a": solved.summary.socp_current_error_max_a,
                "cone_loss_error_w": solved.summary.socp_loss_error_max_w})
    if native.IsMIP:
        row["mip_gap"] = native.MIPGap
        row["bound_chf"] = native.ObjBound
    loss_errors, soc_errors, simultaneous = [], [], []
    for battery in solved.bess:
        previous = battery.soc_init_frac
        for time in solved.periods:
            prediction = battery.result
            charge = float(prediction.charge_kw.iloc[time])
            discharge = float(prediction.discharge_kw.iloc[time])
            reactive = float(prediction.q_kvar.iloc[time])
            physical_loss = battery.q_loss_rated_kw * (reactive / battery.s_max_kva) ** 2
            predicted_loss = float(prediction.inverter_loss_kw.iloc[time])
            loss_errors.append(predicted_loss - physical_loss)
            soc = float(prediction.soc_frac.iloc[time])
            expected = previous + (battery.eta_charge * charge - discharge / battery.eta_discharge - predicted_loss) * solved.dt_h / battery.e_cap_kwh
            soc_errors.append(soc - expected)
            simultaneous.append(min(charge, discharge))
            previous = soc
    row.update({"bess_loss_underestimate_max_w": max(0.0, -min(loss_errors)) * 1000,
                "soc_balance_error_max_frac": max(abs(value) for value in soc_errors),
                "simultaneous_charge_discharge_max_kw": max(simultaneous)})
    return row


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = cases()
    rows = []
    variants = ("original", "strict", "loss", "loss_power", "loss_strict", "continuous", "loss_continuous",
                "base100", "base100_strict", "unit_loss", "base100_unit_loss", "net_grid", "net_grid_loss", "base100_net_grid", "base100_net_grid_loss")
    for name, case in selected.items():
        for variant in variants:
            row = {"case": name, **solve_variant(case, variant)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUTPUT / "ablation.csv", index=False)
            print(json.dumps(row), flush=True)
    (OUTPUT / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
