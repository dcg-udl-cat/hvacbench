from abc import ABC, abstractmethod

from hvacbench.energy_price import EnergyPriceType
from hvacbench.schemas import FloatArray


class BoptestTestcase(ABC):
    """Hardcoded BOPTEST testcase contract used by BoptestEnv."""

    name: str
    base_url: str
    energy_price_type: EnergyPriceType
    step_period_seconds: int

    room_temp: str
    cooling_power: str
    fan_power: str
    heating_power: str

    heating_setpoint: str
    heating_setpoint_activate: str
    cooling_setpoint: str
    cooling_setpoint_activate: str

    outdoor_temp: str
    outdoor_humidity: str
    wind_speed: str
    direct_solar: str

    @abstractmethod
    def required_input_points(self) -> frozenset[str]:
        pass

    @abstractmethod
    def required_measurement_points(self) -> frozenset[str]:
        pass

    @abstractmethod
    def required_forecast_points(self) -> frozenset[str]:
        pass

    @abstractmethod
    def weather_forecast_points(self) -> list[str]:
        pass

    @abstractmethod
    def energy_price_forecast_point(self) -> str:
        pass

    @abstractmethod
    def result_points(self) -> list[str]:
        pass

    @abstractmethod
    def default_control_row(self) -> FloatArray:
        pass
