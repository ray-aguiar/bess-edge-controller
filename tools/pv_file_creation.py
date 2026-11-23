import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Read PV profile data
def read_pv_profile(country):
    PV_PROFILE_DIR = DATA_DIR / country / "pv_profile"
    df = pd.read_csv(next(PV_PROFILE_DIR.glob("*.csv")))

    return df

# Create CSV 
def create_csv(country,case):
    df = read_pv_profile(country)

    # Read PV devices configuration
    PV_CONFIG_DIR = DATA_DIR / country / case
    with open(PV_CONFIG_DIR / "devices.json", "r") as f:
        devices = json.load(f)
    
    df_new = pd.DataFrame()
    df_new["timestamp"] = df.iloc[:, 0]

    # Create columns for each PV device
    for pv in devices["pv"]:
        column_name = pv["profile"].split(":")[1]
        max_kw = pv["p_max_kw"]
        df_new[column_name] = df.iloc[:, 1] * max_kw

    # Save CSV
    df_new.to_csv(PV_CONFIG_DIR / "pv.csv", index=False)

countries = ["london", "switzerland"]
cases = ["5bus", "13bus", "34bus", "123bus"]

for country in countries:
    for case in cases:
        create_csv(country,case)