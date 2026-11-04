"""Select aligned [start, end) intervals; naive dates use Switzerland time."""
import pandas as pd


def configure_pv(case, policy):
    if policy not in {"fixed", "teacher"}:
        raise ValueError("pv_policy must be 'fixed' or 'teacher'")
    if policy == "fixed":
        for device in case.pv:
            device.control, device.power_factor, device.curtailable = "fixed_pf", 1.0, False
    return case


def select_period(case, config, default_steps=96):
    index = case.index
    data_end = index[-1] + pd.Timedelta(hours=case.dt_h)

    def timestamp(value, default):
        if value is None:
            return default
        value = pd.Timestamp(value)
        if index.tz is None:
            if value.tz is not None:
                raise ValueError("Use dates without timezone when dataset timestamps have no timezone")
            return value
        if value.tz is None:
            value = value.tz_localize("Europe/Zurich")
        return value.tz_convert(index.tz)

    start = timestamp(config.get("start_date"), index[0])
    default_end = data_end if default_steps is None else start + pd.Timedelta(hours=default_steps * case.dt_h)
    end = timestamp(config.get("end_date"), default_end)
    boundaries = index.append(pd.DatetimeIndex([data_end]))
    first, last = boundaries.get_indexer([start, end])
    if first < 0 or last <= first:
        raise ValueError("Dates must select a nonempty range within the dataset and align with its intervals")
    period = {"start_date": start.isoformat(), "end_date": end.isoformat(),
              "start_step": int(first), "steps": int(last - first)}
    return case.horizon(int(first), periods=int(last - first)), period
