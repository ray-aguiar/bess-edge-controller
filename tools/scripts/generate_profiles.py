"""Rebuild Swiss consumer tariffs and weather-based PV on the demand index."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pvlib
import requests


ROOT = Path(__file__).resolve().parents[2] / "data" / "5bus-switzerland"
SITE = {"latitude": 47.3769, "longitude": 8.5417, "altitude": 408.0}
TIMEZONE = "Europe/Zurich"
TARIFFS = {2023: 0.272, 2024: 0.3214}  # CHF/kWh, ElCom H4 median, excluding VAT.
TILT = 35.0
AZIMUTH = 180.0
GAMMA_PDC = -0.004
SYSTEM_LOSS = 0.14
INVERTER_EFFICIENCY = 0.96
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_weather(index):
    cache = ROOT / "sources" / "open_meteo_era5.json"
    request_path = cache.with_name("weather_request.json")
    params = {
        "latitude": SITE["latitude"], "longitude": SITE["longitude"],
        "elevation": SITE["altitude"],
        "start_date": (index[0] - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": (index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation,direct_radiation,diffuse_radiation",
        "models": "era5", "timezone": "UTC", "wind_speed_unit": "ms",
    }
    if not cache.exists():
        response = requests.get(WEATHER_API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if "hourly" not in payload:
            raise ValueError("The weather API did not return hourly data")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(response.text, encoding="utf-8")
        request_path.write_text(json.dumps({
            "url": response.url, "params": params,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n", encoding="utf-8")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["params"] != params:
        raise ValueError("Cached weather parameters differ from this scenario")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    weather = pd.DataFrame(payload["hourly"]).set_index("time")
    weather.index = pd.to_datetime(weather.index, utc=True)
    if weather.index.has_duplicates or not np.isfinite(weather.to_numpy()).all():
        raise ValueError("Weather has duplicate times or missing/nonfinite values")
    units = payload["hourly_units"]
    if units["shortwave_radiation"] != "W/m²" or units["wind_speed_10m"] != "m/s":
        raise ValueError("Unexpected weather units")
    return weather, request, payload


def disaggregate(hourly, clear_shape, hour_end):
    """Preserve each hourly mean using a clear-sky shape at five-minute midpoints."""
    shape_mean = clear_shape.groupby(hour_end).transform("mean")
    weights = clear_shape.div(shape_mean.where(shape_mean > 0)).fillna(0.0)
    mean = pd.Series(hourly.reindex(hour_end).to_numpy(), index=clear_shape.index)
    if mean.isna().any():
        raise ValueError("Weather does not cover the demand intervals")
    result = mean * weights
    reconstructed = result.groupby(hour_end).mean()
    expected = hourly.reindex(reconstructed.index)
    daylight = clear_shape.groupby(hour_end).mean() > 0
    if not np.allclose(reconstructed[daylight], expected[daylight], atol=1e-8):
        raise ValueError("Hourly irradiation was not conserved")
    discarded = float(expected[~daylight].sum())
    if discarded > max(1.0, float(expected.sum()) * 0.001):
        raise ValueError("Excessive weather radiation outside the modeled daylight")
    return result, discarded


def solar_profile(index, weather, power_limit_kw):
    # Three midpoint samples approximate each interval's mean power.
    fine_index = pd.date_range(index[0], index[-1] + pd.Timedelta(minutes=15),
                               freq="5min", inclusive="left") + pd.Timedelta(seconds=150)
    hour_end = fine_index.floor("h") + pd.Timedelta(hours=1)
    location = pvlib.location.Location(**SITE, tz=TIMEZONE, name="Zurich reference")
    position = location.get_solarposition(fine_index)
    clear = location.get_clearsky(fine_index, solar_position=position)
    cosine = np.cos(np.radians(position["apparent_zenith"])).clip(lower=0.0)
    beam, discarded_beam = disaggregate(weather["direct_radiation"],
                                        clear["dni"] * cosine, hour_end)
    diffuse, discarded_diffuse = disaggregate(weather["diffuse_radiation"],
                                              clear["dhi"], hour_end)
    dni = beam.div(cosine.where(cosine > 0)).fillna(0.0)
    poa = pvlib.irradiance.get_total_irradiance(
        TILT, AZIMUTH, position["apparent_zenith"], position["azimuth"],
        dni, beam + diffuse, diffuse, model="haydavies", albedo=0.2,
        dni_extra=pvlib.irradiance.get_extra_radiation(fine_index),
    )["poa_global"].clip(lower=0.0)
    # Meteorological temperature/wind are instantaneous; interpolate in UTC.
    meteo = weather[["temperature_2m", "wind_speed_10m"]]
    meteo = meteo.reindex(meteo.index.union(fine_index)).interpolate("time").loc[fine_index]
    cell_temperature = pvlib.temperature.faiman(
        poa, meteo["temperature_2m"], meteo["wind_speed_10m"], u0=25.0, u1=6.84,
    )
    dc_power = pvlib.pvsystem.pvwatts_dc(poa, cell_temperature,
                                        power_limit_kw * 1000.0, GAMMA_PDC)
    dc_power = dc_power.clip(lower=0.0) * (1.0 - SYSTEM_LOSS)
    ac_power = pvlib.inverter.pvwatts(dc_power,
                                     pdc0=power_limit_kw * 1000.0 / INVERTER_EFFICIENCY,
                                     eta_inv_nom=INVERTER_EFFICIENCY)
    ac_power = ac_power.where(position["apparent_elevation"] > 0, 0.0)
    quarter_hour_kw = ac_power.resample("15min").mean().reindex(index) / 1000.0
    if not np.isfinite(quarter_hour_kw).all() or not quarter_hour_kw.between(0, power_limit_kw).all():
        raise ValueError("PV power must be finite and within device limits")
    diagnostics = {
        "discarded_night_beam_wh_per_m2": discarded_beam,
        "discarded_night_diffuse_wh_per_m2": discarded_diffuse,
        "maximum_dni_w_per_m2": float(dni.max()),
    }
    return quarter_hour_kw, diagnostics


def main():
    demand_path = ROOT / "demand.csv"
    demand_hash = file_hash(demand_path)
    timestamp = pd.read_csv(demand_path, usecols=["timestamp"])["timestamp"]
    index = pd.DatetimeIndex(pd.to_datetime(timestamp, utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Demand timestamps must be unique and chronological")
    if not (index[1:] - index[:-1] == pd.Timedelta(minutes=15)).all():
        raise ValueError("Demand must use regular 15-minute intervals")
    if index[0] != index[0].floor("h") or index[-1].minute != 45:
        raise ValueError("This generator expects complete hourly groups")
    years = index.tz_convert(TIMEZONE).year
    if set(years) - set(TARIFFS):
        raise ValueError("A sourced tariff is required for every Swiss calendar year")
    devices = json.loads((ROOT / "devices.json").read_text(encoding="utf-8"))
    pv = next(device for device in devices["pv"] if device["id"] == "pv1")
    if pv["profile"] != "pv.csv:PV5":
        raise ValueError("Unexpected PV profile contract")
    weather, request, raw_weather = load_weather(index)
    power_kw, diagnostics = solar_profile(index, weather, pv["p_max_kw"])
    price = pd.DataFrame({"timestamp": timestamp, "price_per_kwh": [TARIFFS[y] for y in years]})
    generation = pd.DataFrame({"timestamp": timestamp, "PV5": power_kw.to_numpy()})
    price.to_csv(ROOT / "price.csv", index=False, float_format="%.6f", lineterminator="\n")
    generation.to_csv(ROOT / "pv.csv", index=False, float_format="%.6f", lineterminator="\n")
    yearly = {}
    for year in sorted(set(years)):
        selected = years == year
        energy = float(power_kw.iloc[np.flatnonzero(selected)].sum() * 0.25)
        yearly[str(year)] = {
            "rows": int(selected.sum()), "price_chf_per_kwh_excluding_vat": TARIFFS[year],
            "pv_energy_kwh": energy, "pv_yield_kwh_per_kwp": energy / pv["p_max_kw"],
            "pv_peak_kw": float(power_kw.iloc[np.flatnonzero(selected)].max()),
        }
    metadata = {
        "schema_version": 1, "formulation": "single_phase_socp",
        "timestamp_convention": "UTC interval start, copied verbatim from demand.csv",
        "local_timezone": TIMEZONE, "rows": len(index), "dt_h": 0.25,
        "start_utc": index[0].isoformat(),
        "end_exclusive_utc": (index[-1] + pd.Timedelta(minutes=15)).isoformat(),
        "site_assumption": {"name": "Zurich, representative site; demand location not specified", **SITE},
        "pv_model": {
            "library": "pvlib", "version": pvlib.__version__,
            "dc_capacity_kwp": pv["p_max_kw"], "ac_limit_kw": pv["p_max_kw"],
            "tilt_degrees": TILT, "azimuth_degrees": AZIMUTH,
            "gamma_pdc_per_celsius": GAMMA_PDC, "system_loss_fraction": SYSTEM_LOSS,
            "inverter_nominal_efficiency": INVERTER_EFFICIENCY,
            "temperature": "Faiman u0=25, u1=6.84; 10m wind used as proxy for module wind",
            "transposition": "Hay-Davies, albedo=0.2",
            "temporal_model": "hourly beam/diffuse disaggregated with Ineichen clear-sky weights; 5min midpoint samples averaged to 15min",
            "output": "available AC kW before dispatch curtailment; modeled, not measured",
        },
        "weather": {
            "provider": "Open-Meteo", "model": "ECMWF ERA5", "request": request,
            "returned_latitude": raw_weather["latitude"],
            "returned_longitude": raw_weather["longitude"],
            "returned_elevation": raw_weather["elevation"], "diagnostics": diagnostics,
        },
        "tariff": {
            "source": "ElCom 2024 announcement, 2023 comparison updated to 27.20 ct/kWh",
            "consumer_category": "H4, household 4500 kWh/year, national median benchmark",
            "currency": "CHF", "vat_included": False, "intraday_variation": False,
            "export_price": "not calibrated; config.json still uses feed_in_tariff_ratio=1.0",
        },
        "yearly": yearly,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "sha256": {name: file_hash(ROOT / name) for name in (
            "demand.csv", "price.csv", "pv.csv", "../../tools/scripts/generate_profiles.py", "sources/open_meteo_era5.json",
        )},
    }
    if file_hash(demand_path) != demand_hash:
        raise RuntimeError("Demand changed during profile generation")
    (ROOT / "profile_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(yearly, indent=2))


if __name__ == "__main__":
    main()
