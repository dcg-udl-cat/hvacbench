from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict

from hvacbench.boptest.testcase import BoptestTestcase
from hvacbench.schemas import FloatArray


KELVIN_OFFSET = 273.15


class BoptestResponse(BaseModel):
    """Standard BOPTEST response envelope."""

    status: int
    message: str = ""


class Input(BaseModel):
    name: str
    description: str | None = None
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None


class Measurement(Input):
    pass


class ForecastPoint(BaseModel):
    name: str
    description: str | None = None
    unit: str | None = None


class SelectResponse(BoptestResponse):
    payload: dict[str, str]

    @property
    def testid(self) -> str:
        return self.payload["testid"]


class GetInputsResponse(BoptestResponse):
    payload: dict[str, Input]

    @property
    def inputs(self) -> list[Input]:
        return list(self.payload.values())


class GetMeasurementsResponse(BoptestResponse):
    payload: dict[str, Measurement]

    @property
    def measurements(self) -> list[Measurement]:
        return list(self.payload.values())


class GetForecastPointsResponse(BoptestResponse):
    payload: dict[str, ForecastPoint]

    @property
    def forecast_points(self) -> list[ForecastPoint]:
        return list(self.payload.values())


class SetStepResponse(BoptestResponse):
    payload: float | str

    @property
    def step(self) -> float | str:
        return self.payload


class ScenarioPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    electricity_price: str | None = None
    time_period: dict[str, float] | str | None = None
    temperature_uncertainty: str | None = None
    solar_uncertainty: str | None = None
    seed: int | None = None


class SetScenarioResponse(BoptestResponse):
    payload: ScenarioPayload

    @property
    def electricity_price(self) -> str | None:
        return self.payload.electricity_price


class AdvanceResponse(BoptestResponse):
    payload: dict[str, float]

    @property
    def values(self) -> dict[str, float]:
        return self.payload

class InitializeResponse(AdvanceResponse):
    pass


class ForecastResponse(BoptestResponse):
    payload: dict[str, list[float]]

    @property
    def series(self) -> dict[str, list[float]]:
        return self.payload


class ResultsResponse(BoptestResponse):
    payload: dict[str, list[float]]

    @property
    def series(self) -> dict[str, list[float]]:
        return self.payload


class KpiPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    cost_tot: float | None = None
    emis_tot: float | None = None
    ener_tot: float | None = None
    pele_tot: float | None = None
    pgas_tot: float | None = None
    pdih_tot: float | None = None
    idis_tot: float | None = None
    tdis_tot: float | None = None
    time_rat: float | None = None


class KpiResponse(BoptestResponse):
    payload: KpiPayload


class StopPayload(BaseModel):
    stopped: bool = True


class StopResponse(BoptestResponse):
    payload: StopPayload

    @property
    def stopped(self) -> bool:
        return self.payload.stopped


class BaseBoptestClient(ABC):
    """Abstract interface shared by the HTTP and mock BOPTEST clients."""

    testcase: str = "bestest_air"
    testid: str

    @abstractmethod
    def get_inputs(self) -> GetInputsResponse:
        pass

    @abstractmethod
    def get_measurements(self) -> GetMeasurementsResponse:
        pass

    @abstractmethod
    def get_forecast_points(self) -> GetForecastPointsResponse:
        pass

    @abstractmethod
    def set_step(self, step_seconds: int) -> SetStepResponse:
        pass

    @abstractmethod
    def set_scenario(self, electricity_price_type: str) -> SetScenarioResponse:
        pass

    @abstractmethod
    def initialize(
        self,
        start_time_seconds: int,
        warmup_period_seconds: int,
    ) -> InitializeResponse:
        pass

    @abstractmethod
    def advance(self, inputs: dict[str, float]) -> AdvanceResponse:
        pass

    @abstractmethod
    def forecast(
        self,
        point_names: list[str],
        horizon_seconds: int,
        interval_seconds: int,
    ) -> ForecastResponse:
        pass

    @abstractmethod
    def results(
        self,
        point_names: list[str],
        start_time_seconds: int,
        final_time_seconds: int,
    ) -> ResultsResponse:
        pass

    @abstractmethod
    def get_kpis(self) -> KpiResponse:
        pass

    @abstractmethod
    def stop(self) -> StopResponse:
        pass

    def control_row_to_inputs(
        self,
        testcase: BoptestTestcase,
        control_row: FloatArray,
    ) -> dict[str, float]:
        heating_setpoint_c = float(control_row[0])
        cooling_setpoint_c = float(control_row[1])
        return {
            testcase.heating_setpoint: heating_setpoint_c + KELVIN_OFFSET,
            testcase.heating_setpoint_activate: 1.0,
            testcase.cooling_setpoint: cooling_setpoint_c + KELVIN_OFFSET,
            testcase.cooling_setpoint_activate: 1.0,
        }

    def advance_control_row(
        self,
        testcase: BoptestTestcase,
        control_row: FloatArray,
    ) -> AdvanceResponse:
        return self.advance(self.control_row_to_inputs(testcase, control_row))

    def extract_state_from_values(
        self,
        testcase: BoptestTestcase,
        values: dict[str, float],
    ) -> FloatArray:
        missing = testcase.required_measurement_points().difference(values)
        if missing:
            msg = (
                "BOPTEST payload is missing required bestest_air measurement "
                f"point(s): {sorted(missing)}."
            )
            raise KeyError(msg)

        room_temp_c = float(values[testcase.room_temp]) - KELVIN_OFFSET
        hvac_power_w = (
            float(values[testcase.cooling_power])
            + float(values[testcase.fan_power])
            + float(values[testcase.heating_power])
        )
        return np.array([room_temp_c, hvac_power_w], dtype=np.float64)

    def state_rows_from_results(
        self,
        testcase: BoptestTestcase,
        response: ResultsResponse,
    ) -> FloatArray:
        for point_name in testcase.required_measurement_points():
            if point_name not in response.series:
                raise KeyError(f"BOPTEST results payload missing {point_name!r}.")

        room_temp_c = self._series(response, testcase.room_temp) - KELVIN_OFFSET
        hvac_power_w = (
            self._series(response, testcase.cooling_power)
            + self._series(response, testcase.fan_power)
            + self._series(response, testcase.heating_power)
        )
        return np.column_stack([room_temp_c, hvac_power_w]).astype(np.float64)

    def convert_weather_point(
        self,
        testcase: BoptestTestcase,
        point_name: str,
        values: np.ndarray,
    ) -> np.ndarray:
        if point_name == testcase.outdoor_temp:
            return values - KELVIN_OFFSET
        if point_name == testcase.outdoor_humidity:
            return values * 100.0
        return values

    @staticmethod
    def _series(
        response: ForecastResponse | ResultsResponse,
        point_name: str,
    ) -> np.ndarray:
        if point_name not in response.series:
            raise KeyError(f"BOPTEST payload missing point {point_name!r}.")
        return np.asarray(response.series[point_name], dtype=np.float64)
