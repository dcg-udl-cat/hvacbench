from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the digital twin environment dimensions and variables."""
    history_length: int = 1536
    horizon: int = 96
    
    state_variables: Tuple[str, ...] = (
        "Room Air Temperature (C)",
        "HVAC Power Consumption (W)",
    )
    
    weather_variables: Tuple[str, ...] = (
        "Outdoor Air Temperature (C)",
        "Outdoor Humidity (%)",
        "Wind Speed (m/s)",
        "Direct Solar Radiation (W/m^2)",
    )
    
    control_variables: Tuple[str, ...] = (
        "Heating Setpoint (C)",
        "Cooling Setpoint (C)",
    )

    timestep_minutes: int = 15
    timestamp_column: str = "time"
    frequency: str = "15min"

    @property
    def n_states(self) -> int:
        return len(self.state_variables)

    @property
    def n_weather(self) -> int:
        return len(self.weather_variables)

    @property
    def n_controls(self) -> int:
        return len(self.control_variables)
