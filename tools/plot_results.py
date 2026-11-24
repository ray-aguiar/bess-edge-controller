"""Export full-period and daily plots from test CSV results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parameters
TRAINING = "supervised"
ARCHITECTURE = "mlp"
SEED = 42
TRIAL = 0
DAY = "2024-01-03"

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "test" / TRAINING / ARCHITECTURE / f"seed_{SEED}" / f"trial_{TRIAL}"
OUTPUT_DIR = INPUT_DIR / "plots"

COLORS = ["dodgerblue", "gold", "hotpink", "blueviolet", "seagreen", "darkorange", "deeppink"]

def aggregate_buses(buses, suffix):
    columns = [c for c in buses.columns if c.startswith("bus_") and c.endswith(suffix)]
    if not columns:
        raise ValueError(f"Nenhuma coluna encontrada para o sufixo {suffix}.")
    result = buses[["timestamp"]].copy()
    result["total"] = buses[columns].sum(axis=1)
    return result.groupby("timestamp", as_index=False)["total"].sum().sort_values("timestamp")

def load_data():
    data = {name: pd.read_csv(INPUT_DIR / f"{name}.csv") for name in ("bess", "buses", "grid", "pv")}
    for name, df in data.items():
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        data[name] = df.sort_values("timestamp").reset_index(drop=True)
    return data

def get_interval(grid):
    intervals = grid["timestamp"].drop_duplicates().sort_values().diff().dropna().dt.total_seconds()
    if intervals.empty:
        raise ValueError("Não foi possível determinar o intervalo de simulação.")
    return intervals.median() / 3600

def select_day(df):
    return df[df["timestamp"].dt.date == pd.Timestamp(DAY).date()].copy()

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

def save_figure(fig, ax, filename):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)

def plot_soc(bess, dt, daily=False):
    fig, ax = setup_figure("Battery state of charge", "State of charge", daily)
    groups = bess.groupby("device") if "device" in bess.columns else [("BESS", bess)]

    for i, (device, df) in enumerate(groups):
        df = df.sort_values("timestamp")
        plot_step(ax, elapsed_hours(df, daily), df["soc_before_frac"], dt,
                  color=COLORS[i % len(COLORS)], label=str(device))

    ax.axhline(0.9, color="black", linestyle="--", label="Upper limit")
    ax.axhline(0.1, color="black", linestyle=":", label="Lower limit")
    ax.set_ylim(0, 1)
    save_figure(fig, ax, "daily_soc.png" if daily else "full_soc.png")

def plot_voltages(buses, dt, daily=False):
    fig, ax = setup_figure("Bus voltage magnitude", "Voltage [pu]", daily)
    columns = sorted(c for c in buses.columns if c.startswith("bus_") and c.endswith("_v_pu"))

    for i, column in enumerate(columns):
        df = buses.sort_values("timestamp")
        plot_step(ax, elapsed_hours(df, daily), df[column], dt,
                  color=COLORS[i % len(COLORS)], linewidth=1, label=column.replace("_v_pu", ""))

    save_figure(fig, ax, "daily_voltages.png" if daily else "full_voltages.png")

def plot_cost(grid, dt, daily=False):
    fig, ax = setup_figure("Energy cost per interval", "Cost [CHF]", daily)
    df = grid.sort_values("timestamp")
    plot_step(ax, elapsed_hours(df, daily), df["cost_chf"], dt, color="hotpink", label="Cost")
    save_figure(fig, ax, "daily_cost.png" if daily else "full_cost.png")

def aggregate(df, column):
    return df.groupby("timestamp", as_index=False)[column].sum().sort_values("timestamp")

def plot_power_flow(pv, grid, bess, buses, dt):
    fig, ax = setup_figure("Active power flow", "Power [kW]", daily=True)
    ax.axhline(0, color="black", linestyle="-.")

    load = aggregate_buses(buses, "_p_injection_kw")
    plot_step(ax, elapsed_hours(load, daily=True), load["total"], dt, color="black", label="Load")

    for df, column, color, label in (
        (pv, "generation_kw", "gold", "PV generation"),
        (grid, "p_import_kw", "hotpink", "Grid import"),
    ):
        df = aggregate(df, column)
        plot_step(ax, elapsed_hours(df, daily=True), df[column], dt, color=color, label=label)

    bess = aggregate(bess, "p_net_kw")
    hours = elapsed_hours(bess, daily=True)
    ax.bar(hours, bess["p_net_kw"], width=0.9 * dt, align="edge", color="blueviolet", label="BESS")

    save_figure(fig, ax, "daily_power_flow.png")

def plot_reactive_power(pv, grid, bess, buses, dt):
    fig, ax = setup_figure("Reactive power flow", "Reactive power [kvar]", daily=True)
    ax.axhline(0, color="black", linestyle="-.")

    load = aggregate_buses(buses, "_q_injection_kvar")
    plot_step(ax, elapsed_hours(load, daily=True), load["total"], dt, color="black", label="Load")

    for df, column, color, label in (
        (pv, "q_injection_kvar", "gold", "PV"),
        (grid, "q_injection_kvar", "hotpink", "Grid"),
    ):
        df = aggregate(df, column)
        plot_step(ax, elapsed_hours(df, daily=True), df[column], dt, color=color, label=label)

    bess = aggregate(bess, "q_injection_kvar")
    hours = elapsed_hours(bess, daily=True)
    ax.bar(hours, bess["q_injection_kvar"], width=0.9 * dt, align="edge", color="blueviolet", label="BESS")

    save_figure(fig, ax, "daily_reactive_power.png")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    dt = get_interval(data["grid"])

    plot_soc(data["bess"], dt)
    plot_voltages(data["buses"], dt)
    plot_cost(data["grid"], dt)

    daily = {name: select_day(df) for name, df in data.items()}
    if any(df.empty for df in daily.values()):
        raise ValueError(f"Não foram encontrados dados para o dia {DAY}.")

    plot_soc(daily["bess"], dt, daily=True)
    plot_voltages(daily["buses"], dt, daily=True)
    plot_cost(daily["grid"], dt, daily=True)
    plot_power_flow(daily["pv"], daily["grid"], daily["bess"], daily["buses"], dt)
    plot_reactive_power(daily["pv"], daily["grid"], daily["bess"], daily["buses"], dt)

    print(f"Plots saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
