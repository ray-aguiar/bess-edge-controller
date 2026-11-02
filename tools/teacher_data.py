"""Teacher-only trajectories with causal observations from the previous solve."""
from opf_teacher.opf.single_phase.states import BessState, BusState, PvState
from opf_teacher.teacher import PairSelector, StateActionHistory, Teacher


def configure_pv(case, policy):
    if policy not in {"fixed", "teacher"}:
        raise ValueError("pv_policy must be 'fixed' or 'teacher'")
    if policy == "fixed":
        for device in case.pv:
            device.control, device.power_factor, device.curtailable = "fixed_pf", 1.0, False
    return case


def rollout(case, horizon=None):
    buses = {bid: BusState(case.grid.v_ref_pu, 0.0, 0.0) for bid in case.buses}
    bess = {device.id: BessState(device.soc_init_frac, 0.0, 0.0) for device in case.bess}
    pv = {device.id: PvState(0.0, 0.0, 0.0) for device in case.pv}
    for step in case.periods:
        for bid, bus in case.buses.items():
            buses[bid].p_load_kw = float(bus.p_load_kw.iloc[step])
            buses[bid].q_load_kvar = float(bus.q_load_kw.iloc[step])
        for device in case.pv:
            pv[device.id].available_kw = float(device.avail_kw.iloc[step])
        periods = None if horizon is None else min(horizon, case.n_periods - step)
        teacher = Teacher(case.horizon(step, periods=periods), formulation="single_phase_socp")
        teacher.observe(buses=buses, bess=bess, pv=pv)
        solved = teacher.solve()
        yield teacher
        for bid, bus in solved.buses.items():
            buses[bid].v_before_pu = float(bus.result.v_pu.iloc[0])
        for device in solved.bess:
            state = bess[device.id]
            state.soc_before_frac = float(device.result.soc_frac.iloc[0])
            state.previous_p_kw = float(device.result.p_net_kw.iloc[0])
            state.previous_q_kvar = float(device.result.q_kvar.iloc[0])
        for device in solved.pv:
            pv[device.id].previous_p_kw = float(device.result.gen_kw.iloc[0])
            pv[device.id].previous_q_kvar = float(device.result.q_kvar.iloc[0])


def collect_pairs(case, config):
    """Collect valid local observation windows and current BESS actions."""
    configure_pv(case, "fixed")
    selector = PairSelector(case, mode="local", device_id=config["device"])
    history = StateActionHistory(window=config["window"])
    x, y = [], []
    for teacher in rollout(case, horizon=config["horizon"]):
        state, action = teacher.state_action()
        local_x, local_y = selector.select(state, action)
        pair = history.append(local_x, local_y)
        if pair is not None and teacher.case.quality["valid_for_imitation"]:
            x.append(pair[0])
            y.append(pair[1])
    if not x:
        raise ValueError("No valid pairs; check the dates, window and teacher quality")
    return x, y
