from dataclasses import dataclass
from .float_array import FloatArray

@dataclass(frozen=True)
class Observation:
    """Represents a full observation at a step, containing current histories and future forecasts."""
    weather_history: FloatArray
    control_history: FloatArray
    state_history: FloatArray
    weather_forecast: FloatArray
    energy_price_forecast: FloatArray
