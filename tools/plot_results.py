
"""Export full-period and daily plots from test CSV results."""
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["dodgerblue", "gold", "hotpink", "blueviolet", "seagreen", "darkorange", "deeppink"]


def aggregate_buses(buses, suffix):
    columns = [c for c in buses.columns if c.startswith("bus_") and c.endswith(suffix)]
    if not columns:
        raise ValueError(f"No column found for the suffix {suffix}.")
    result = buses[["timestamp"]].copy()
    result["total"] = buses[columns].sum(axis=1)
    return result.groupby("timestamp", as_index=False)["total"].sum().sort_values("timestamp")


def load_data(input_dir):
    input_dir = Path(input_dir)
    data = {name: pd.read_csv(input_dir / f"{name}.csv")
            for name in ("bess", "buses", "grid", "pv")}

    for name, df in data.items():
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        data[name] = df.sort_values("timestamp").reset_index(drop=True)

    return data


def get_interval(grid):
    intervals = grid["timestamp"].drop_duplicates().sort_values().diff().dropna().dt.total_seconds()
    if intervals.empty:
        raise ValueError("It wasn't possible to determine the simulation interval.")
    return intervals.median() / 3600


def select_day(df, day):
    return df[df["timestamp"].dt.date == pd.Timestamp(day).date()].copy()


def elapsed_hours(df, daily=False):
    if daily:
        return (df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
                + df["timestamp"].dt.second / 3600).to_numpy()
    return (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 3600


def plot_step(ax, hours, values, dt, **style):
    hours, values = np.asarray(hours), np.asarray(values)
    if len(hours):
        ax.step(np.append(hours, hours[-1] + dt), np.append(values, values[-1]), where="post", **style)


def setup_figure(title, ylabel, daily=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set(title=title, xlabel="Time of day [h]" if daily else "Elapsed time [h]", ylabel=ylabel)
    if daily:
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 2))
    ax.grid(color="lightgrey")
    ax.set_axisbelow(True)
    return fig, ax


def save_figure(fig, ax, filename, output_dir):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def load_soc_limits(case):
    """Load each BESS's SOC limits from its case directory."""
    devices_path = Path(case) / "devices.json"
    devices = json.loads(devices_path.read_text(encoding="utf-8"))
    return {
        device["id"]: (
            device.get("soc_min_frac", 0.0),
            device.get("soc_max_frac", 1.0),
        )
        for device in devices["bess"]
    }


def plot_soc(bess, dt, soc_limits, daily=False, output_dir=None):
    fig, ax = setup_figure("Battery state of charge", "State of charge", daily)
    groups = bess.groupby("device") if "device" in bess.columns else [("BESS", bess)]

    for i, (device, df) in enumerate(groups):
        df = df.sort_values("timestamp")
        if device not in soc_limits:
            raise ValueError(f"No SOC limits found for BESS {device!r}")
        plot_step(ax, elapsed_hours(df, daily), df["soc_before_frac"], dt,
                  color=COLORS[i % len(COLORS)], label=str(device))
        soc_min, soc_max = soc_limits[device]
        color = COLORS[i % len(COLORS)]
        ax.axhline(soc_max, color=color, linestyle="--", label=f"{device} upper limit")
        ax.axhline(soc_min, color=color, linestyle=":", label=f"{device} lower limit")
    ax.set_ylim(0, 1)
    save_figure(fig, ax, "daily_soc.png" if daily else "full_soc.png", output_dir)

def plot_bess_power(bess, dt, daily=False, output_dir=None):
    fig, ax = setup_figure("BESS active power", "Power [kW]", daily)
    ax.axhline(0, color="black", linestyle="-.")

    df = aggregate(bess, "p_requested_kw")
    plot_step(ax, elapsed_hours(df, daily), df["p_requested_kw"], dt,
              color="blueviolet", label="Requested power")

    df = aggregate(bess, "p_net_kw")
    plot_step(ax, elapsed_hours(df, daily), df["p_net_kw"], dt,
              color="hotpink", label="Net power")

    filename = "daily_bess_power.png" if daily else "full_bess_power.png"
    save_figure(fig, ax, filename, output_dir)

def plot_voltages(buses, dt, daily=False, output_dir=None):
    fig, ax = setup_figure("Bus voltage magnitude", "Voltage [pu]", daily)
    columns = {}
    phase_order = {"a": 0, "b": 1, "c": 2, None: 3}
    for column in buses.columns:
        match = re.fullmatch(r"(bus_.+?)(?:_([abc]))?_v_pu", column)
        if match:
            bus, phase = match.groups()
            if bus not in columns or phase_order[phase] < phase_order[columns[bus][0]]:
                columns[bus] = (phase, column)

    for i, (bus, (_, column)) in enumerate(sorted(columns.items())):
        df = buses.sort_values("timestamp")
        plot_step(ax, elapsed_hours(df, daily), df[column], dt,
                  color=COLORS[i % len(COLORS)], linewidth=1,
                  label=bus)

    save_figure(fig, ax, "daily_voltages.png" if daily else "full_voltages.png", output_dir)


def plot_cost(grid, dt, daily=False, output_dir=None):
    fig, ax = setup_figure("Energy cost per interval", "Cost [CHF]", daily)
    df = grid.sort_values("timestamp")
    plot_step(ax, elapsed_hours(df, daily), df["cost_chf"], dt,
              color="hotpink", label="Cost")
    save_figure(fig, ax, "daily_cost.png" if daily else "full_cost.png", output_dir)


def aggregate(df, column):
    return df.groupby("timestamp", as_index=False)[column].sum().sort_values("timestamp")


def plot_power_flow(pv, grid, bess, buses, dt, output_dir=None):
    fig, ax = setup_figure("Active power flow", "Power [kW]", daily=True)
    ax.axhline(0, color="black", linestyle="-.")

    grid = grid.copy()
    grid["p_net_grid_kw"] = grid["p_import_kw"] - grid["p_export_kw"]

    for df, column, color, label in (
        (pv, "generation_kw", "gold", "PV generation"),
        (grid, "p_net_grid_kw", "hotpink", "Grid"),
    ):
        df = aggregate(df, column)
        plot_step(ax, elapsed_hours(df, daily=True), df[column], dt,
                  color=color, label=label)

    bess = aggregate(bess, "p_net_kw")
    hours = elapsed_hours(bess, daily=True)
    ax.bar(hours, bess["p_net_kw"], width=0.9 * dt, align="edge",
           color="blueviolet", label="BESS")

    save_figure(fig, ax, "daily_power_flow.png", output_dir)


def plot_reactive_power(pv, grid, bess, buses, dt, output_dir=None):
    fig, ax = setup_figure("Reactive power flow", "Reactive power [kvar]", daily=True)
    ax.axhline(0, color="black", linestyle="-.")

    for df, column, color, label in (
        (pv, "q_injection_kvar", "gold", "PV"),
        (grid, "q_injection_kvar", "hotpink", "Grid"),
    ):
        df = aggregate(df, column)
        plot_step(ax, elapsed_hours(df, daily=True), df[column], dt,
                  color=color, label=label)

    bess = aggregate(bess, "q_injection_kvar")
    hours = elapsed_hours(bess, daily=True)
    ax.bar(hours, bess["q_injection_kvar"], width=0.9 * dt, align="edge",
           color="blueviolet", label="BESS")

    save_figure(fig, ax, "daily_reactive_power.png", output_dir)


def create_plots(input_dir, day, output_dir=None):
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir / "plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(input_dir)
    report = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    soc_limits = load_soc_limits(report["config"]["case"])
    dt = get_interval(data["grid"])

    # Full-period plots
    plot_soc(data["bess"], dt, soc_limits, output_dir=output_dir)
    plot_voltages(data["buses"], dt, output_dir=output_dir)
    plot_cost(data["grid"], dt, output_dir=output_dir)
    plot_bess_power(data["bess"], dt, output_dir=output_dir)

    # Daily plots
    daily = {name: select_day(df, day) for name, df in data.items()}

    if any(df.empty for df in daily.values()):
        raise ValueError(f"Data wasn't found for date {day}.")

    plot_soc(daily["bess"], dt, soc_limits, daily=True, output_dir=output_dir)
    plot_bess_power(daily["bess"], dt, daily = True, output_dir=output_dir)
    plot_voltages(daily["buses"], dt, daily=True, output_dir=output_dir)
    plot_cost(daily["grid"], dt, daily=True, output_dir=output_dir)
    plot_power_flow(daily["pv"], daily["grid"], daily["bess"],
                    daily["buses"], dt, output_dir=output_dir)
    plot_reactive_power(daily["pv"], daily["grid"], daily["bess"],
                        daily["buses"], dt, output_dir=output_dir)
