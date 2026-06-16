from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from hvacbench.boptest.testcase import BoptestTestcase
from hvacbench.energy_price import EnergyPriceType


PRICE_FORECAST_POINTS: dict[EnergyPriceType, str] = {
    EnergyPriceType.CONSTANT: "PriceElectricPowerConstant",
    EnergyPriceType.DYNAMIC: "PriceElectricPowerDynamic",
    EnergyPriceType.HIGHLY_DYNAMIC: "PriceElectricPowerHighlyDynamic",
}


@dataclass(frozen=True)
class BestestAir(BoptestTestcase):
    """Hardcoded BOPTEST mapping for the bestest_air testcase."""

    base_url: str = "http://127.0.0.1"
    energy_price_type: EnergyPriceType = EnergyPriceType.DYNAMIC

    name: str = "bestest_air"
    step_period_seconds: int = 900

    room_temp: str = "zon_reaTRooAir_y"
    cooling_power: str = "fcu_reaPCoo_y"
    fan_power: str = "fcu_reaPFan_y"
    heating_power: str = "fcu_reaPHea_y"

    heating_setpoint: str = "con_oveTSetHea_u"
    heating_setpoint_activate: str = "con_oveTSetHea_activate"
    cooling_setpoint: str = "con_oveTSetCoo_u"
    cooling_setpoint_activate: str = "con_oveTSetCoo_activate"

    outdoor_temp: str = "TDryBul"
    outdoor_humidity: str = "relHum"
    wind_speed: str = "winSpe"
    direct_solar: str = "HDirNor"

    default_heating_setpoint_c: float = 21.0
    default_cooling_setpoint_c: float = 24.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "energy_price_type",
            EnergyPriceType(self.energy_price_type),
        )

    def required_input_points(self) -> frozenset[str]:
        return frozenset(
            {
                self.heating_setpoint,
                self.heating_setpoint_activate,
                self.cooling_setpoint,
                self.cooling_setpoint_activate,
            }
        )

    def required_measurement_points(self) -> frozenset[str]:
        return frozenset(
            {
                self.room_temp,
                self.cooling_power,
                self.fan_power,
                self.heating_power,
            }
        )

    def required_forecast_points(self) -> frozenset[str]:
        return frozenset(
            {
                self.outdoor_temp,
                self.outdoor_humidity,
                self.wind_speed,
                self.direct_solar,
                *PRICE_FORECAST_POINTS.values(),
            }
        )

    def weather_forecast_points(self) -> list[str]:
        return [
            self.outdoor_temp,
            self.outdoor_humidity,
            self.wind_speed,
            self.direct_solar,
        ]

    def energy_price_forecast_point(self) -> str:
        return PRICE_FORECAST_POINTS[self.energy_price_type]

    def result_points(self) -> list[str]:
        return [
            self.room_temp,
            self.cooling_power,
            self.fan_power,
            self.heating_power,
        ]

    def default_control_row(self) -> Float[np.ndarray, "2"]:
        return np.array(
            [self.default_heating_setpoint_c, self.default_cooling_setpoint_c],
            dtype=np.float64,
        )
