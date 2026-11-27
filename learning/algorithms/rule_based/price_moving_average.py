"""Price-threshold BESS dispatch using a rolling moving average."""
from collections import deque
import math

from opf_teacher.opf.single_phase import load_case
from opf_teacher.teacher import select_extra_features
from tools.dataset import select_period


WINDOW = 96
POWER_FRACTION = 0.2
EXTRA_FEATURES = {"grid": ["buy_price_01"]}


class PriceMovingAverageStrategy:
    """Charge below the recent mean price and discharge above it."""

    def __init__(self, window, power_fraction=0.2, initial_prices=()):
        self.window = window
        self.power_fraction = power_fraction
        self.initial_prices = tuple(initial_prices)
        self.prices = deque(self.initial_prices, maxlen=window)

    def reset(self):
        self.prices.clear()
        self.prices.extend(self.initial_prices)

    def action(self, state, p_charge_max_pu, p_discharge_max_pu):
        """Return P/Q in pu; positive P charges and negative P discharges."""
        price = float(state["extras"]["grid"]["buy_price_01"])

        active_power = 0.0
        if self.prices:
            average = sum(self.prices) / len(self.prices)
            if not math.isclose(price, average, rel_tol=0.0, abs_tol=1e-12):
                if price < average:
                    active_power = self.power_fraction * p_charge_max_pu
                elif price > average:
                    active_power = -self.power_fraction * p_discharge_max_pu
        self.prices.append(price)
        return active_power, 0.0


def create_strategy(config):
    """Create the strategy with recent prices from before the evaluation period."""
    case = load_case(config["case"])
    _, period = select_period(case, config, default_steps=None)
    start_step = period["start_step"]
    if start_step < WINDOW:
        raise ValueError(f"Need {WINDOW} earlier price steps before the selected start date")
    prices = case.price.iloc[start_step - WINDOW:start_step]
    initial_prices = [
        select_extra_features({"grid": {"buy_price_per_kwh": price}}, None, EXTRA_FEATURES)
        ["grid"]["buy_price_01"]
        for price in prices
    ]
    return PriceMovingAverageStrategy(WINDOW, POWER_FRACTION, initial_prices), {
        "window": WINDOW,
        "power_fraction": POWER_FRACTION,
        "price_history": {
            "steps": len(initial_prices),
            "start": case.index[start_step - WINDOW].isoformat(),
            "end_exclusive": case.index[start_step].isoformat(),
        },
    }