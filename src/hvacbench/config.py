from dataclasses import dataclass, field


STEP_PERIOD_SECONDS = 900
TIMESTEP_MINUTES = 15
N_STATES = 2
N_WEATHER = 4
N_CONTROLS = 2


@dataclass(frozen=True)
class EnvConfig:
    """Shared environment sizing and episode configuration."""

    history_length: int = 1536
    horizon: int = 96
    total_simulation_seconds: int = 14 * 24 * 3600

    @property
    def timestep_minutes(self) -> int:
        return TIMESTEP_MINUTES

    @property
    def step_period_seconds(self) -> int:
        return STEP_PERIOD_SECONDS

    @property
    def n_states(self) -> int:
        return N_STATES

    @property
    def n_weather(self) -> int:
        return N_WEATHER

    @property
    def n_controls(self) -> int:
        return N_CONTROLS


@dataclass(frozen=True)
class TTMStateVars:
    temp: str = "Room Air Temperature (C)"
    power: str = "HVAC Power Consumption (W)"

    @property
    def names(self) -> tuple[str, str]:
        return self.temp, self.power


@dataclass(frozen=True)
class TTMWeatherVars:
    outdoor_temp: str = "Outdoor Air Temperature (C)"
    outdoor_humidity: str = "Outdoor Humidity (%)"
    wind_speed: str = "Wind Speed (m/s)"
    direct_solar: str = "Direct Solar Radiation (W/m^2)"

    @property
    def names(self) -> tuple[str, str, str, str]:
        return (
            self.outdoor_temp,
            self.outdoor_humidity,
            self.wind_speed,
            self.direct_solar,
        )


@dataclass(frozen=True)
class TTMControlVars:
    heating_setpoint: str = "Heating Setpoint (C)"
    cooling_setpoint: str = "Cooling Setpoint (C)"

    @property
    def names(self) -> tuple[str, str]:
        return self.heating_setpoint, self.cooling_setpoint


@dataclass(frozen=True)
class TTMVariables:
    state_vars: TTMStateVars = field(default_factory=TTMStateVars)
    weather_vars: TTMWeatherVars = field(default_factory=TTMWeatherVars)
    control_vars: TTMControlVars = field(default_factory=TTMControlVars)

    @property
    def state_names(self) -> tuple[str, str]:
        return self.state_vars.names

    @property
    def weather_names(self) -> tuple[str, str, str, str]:
        return self.weather_vars.names

    @property
    def control_names(self) -> tuple[str, str]:
        return self.control_vars.names
