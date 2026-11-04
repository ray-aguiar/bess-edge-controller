"""Export executed electrical measurements with explicit physical units."""
from pathlib import Path

import pandas as pd


def save_timeseries(records, output):
    tables = {name: [] for name in ("buses", "branches", "bess", "pv", "grid")}
    for record in records:
        info = record["info"]
        time = {"timestamp": info["timestamp"]}
        base = info["base"]["s_base_kva"]
        flows = info["branch_measurements"]
        single_phase = all(len(phases) == 1 for phases in info["bus_voltages_pu"].values())
        buses, branches = dict(time), dict(time)
        for flow in flows:
            prefix = f"{flow['element'].lower().replace('.', '_')}_t{flow['terminal']}"
            if not single_phase:
                prefix += f"_{flow['phase']}"
            values = {
                "p_kw": flow["p_flow_pu"] * base, "q_kvar": flow["q_flow_pu"] * base,
                "current_a": flow["current_a"], "current_angle_deg": flow["current_angle_deg"],
            }
            branches.update({f"{prefix}_{name}": value for name, value in values.items()})
        for bus, phases in info["bus_voltages_pu"].items():
            for phase, voltage in phases.items():
                connected = [flow for flow in flows if flow["bus"] == bus and flow["phase"] == phase]
                prefix = bus if single_phase else f"{bus}_{phase}"
                values = {
                    "v_pu": voltage,
                    "v_v": info["bus_voltages_v"][bus][phase],
                    "angle_deg": info["bus_angles_deg"][bus][phase],
                    "p_injection_kw": sum(flow["p_flow_pu"] for flow in connected) * base,
                    "q_injection_kvar": sum(flow["q_flow_pu"] for flow in connected) * base,
                }
                buses.update({f"{prefix}_{name}": value for name, value in values.items()})
        tables["buses"].append(buses)
        tables["branches"].append(branches)
        for device, values in info["executed_action"]["bess"].items():
            measured = info["device_measurements"]["bess"][device]
            requested = info["requested_action"]["bess"][device]
            p, q = requested["p_net_pu"], requested.get("q_injection_pu", 0.0)
            tables["bess"].append(time | {
                "device": device,
                "p_requested_kw": (sum(p.values()) if isinstance(p, dict) else p) * base,
                "q_requested_kvar": (sum(q.values()) if isinstance(q, dict) else q) * base,
                "p_net_kw": measured["p_net_total_pu"] * base,
                "q_injection_kvar": measured["q_injection_total_pu"] * base,
                "soc_before_frac": values["soc_before_frac"], "soc_after_frac": values["soc_after_frac"],
            })
        for device, values in info["device_measurements"]["pv"].items():
            tables["pv"].append(time | {
                "device": device, "generation_kw": values["generation_total_pu"] * base,
                "q_injection_kvar": values["q_injection_total_pu"] * base,
                "available_kw": info["executed_action"]["pv"][device]["available_pu"] * base,
            })
        tables["grid"].append(time | {
            "p_import_kw": info["grid_import_pu"] * base,
            "p_export_kw": info["grid_export_pu"] * base,
            "q_injection_kvar": info["grid_q_pu"] * base,
            "cost_chf": info["cost"], "reward": info["reward"],
        })
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
