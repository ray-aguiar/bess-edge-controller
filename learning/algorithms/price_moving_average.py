"""Price-threshold BESS dispatch using a rolling moving average."""
from collections import deque
import math

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