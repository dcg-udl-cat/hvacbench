from typing import Any

import numpy as np

from hvacbench.boptest.base import (
    BaseBoptestClient,
    ForecastPoint,
    ForecastResponse,
    GetForecastPointsResponse,
    GetInputsResponse,
    GetMeasurementsResponse,
    Input,
    KpiResponse,
    KpiPayload,
    Measurement,
    AdvanceResponse,
    ResultsResponse,
    ScenarioPayload,
    SetScenarioResponse,
    SetStepResponse,
    StopPayload,
    StopResponse, InitializeResponse,
)
from hvacbench.boptest.bestest_air import BestestAir


class MockBoptestClient(BaseBoptestClient):
    """Deterministic BOPTEST client for tests and examples that need no server."""

    def __init__(
        self,
        testid: str = "mock-testid",
        testcase: str = "bestest_air",
        missing_inputs: set[str] | None = None,
        missing_measurements: set[str] | None = None,
        missing_forecasts: set[str] | None = None,
    ) -> None:
        self.testcase = testcase
        self.testid = testid
        self.current_time_seconds = 0
        self.advance_calls: list[dict[str, float]] = []
        self.initialize_calls: list[dict[str, int]] = []
        self.forecast_calls: list[dict[str, Any]] = []
        self.results_calls: list[dict[str, Any]] = []
        self.stop_calls = 0
        self.scenario = ""
        self.step_seconds = 900
        self.testcase_definition = BestestAir()
        self._missing_inputs = missing_inputs or set()
        self._missing_measurements = missing_measurements or set()
        self._missing_forecasts = missing_forecasts or set()

    def get_inputs(self) -> GetInputsResponse:
        points = self.testcase_definition.required_input_points()
        inputs = [Input(name=point) for point in points.difference(self._missing_inputs)]
        payload = {input_.name: input_ for input_ in inputs}
        return GetInputsResponse(
            status=200,
            message="",
            payload=payload,
        )

    def get_measurements(self) -> GetMeasurementsResponse:
        points = self.testcase_definition.required_measurement_points()
        measurements = [
            Measurement(name=point)
            for point in points.difference(self._missing_measurements)
        ]
        payload = {measurement.name: measurement for measurement in measurements}
        return GetMeasurementsResponse(
            status=200,
            message="",
            payload=payload,
        )

    def get_forecast_points(self) -> GetForecastPointsResponse:
        points = self.testcase_definition.required_forecast_points()
        forecast_points = [
            ForecastPoint(name=point)
            for point in points.difference(self._missing_forecasts)
        ]
        payload = {forecast_point.name: forecast_point for forecast_point in forecast_points}
        return GetForecastPointsResponse(
            status=200,
            message="",
            payload=payload,
        )

    def set_step(self, step_seconds: int) -> SetStepResponse:
        self.step_seconds = step_seconds
        return SetStepResponse(
            status=200,
            message="",
            payload=float(step_seconds),
        )

    def set_scenario(self, electricity_price_type: str) -> SetScenarioResponse:
        self.scenario = electricity_price_type
        return SetScenarioResponse(
            status=200,
            message="",
            payload=ScenarioPayload(electricity_price=electricity_price_type),
        )

    def initialize(
        self,
        start_time_seconds: int,
        warmup_period_seconds: int,
    ) -> InitializeResponse:
        self.current_time_seconds = start_time_seconds + warmup_period_seconds
        self.advance_calls = []
        self.initialize_calls.append(
            {
                "start_time_seconds": start_time_seconds,
                "warmup_period_seconds": warmup_period_seconds,
            }
        )
        payload = self._payload()
        return InitializeResponse(status=200, message="", payload=payload)

    def advance(self, inputs: dict[str, float]) -> AdvanceResponse:
        self.advance_calls.append(dict(inputs))
        self.current_time_seconds += self.step_seconds
        payload = self._payload()
        return AdvanceResponse(status=200, message="", payload=payload)

    def forecast(
        self,
        point_names: list[str],
        horizon_seconds: int,
        interval_seconds: int,
    ) -> ForecastResponse:
        self.forecast_calls.append(
            {
                "point_names": point_names,
                "horizon_seconds": horizon_seconds,
                "interval_seconds": interval_seconds,
            }
        )
        count = horizon_seconds // interval_seconds + 1
        index = np.arange(count, dtype=np.float64)
        data = {
            "TDryBul": 273.15 + 10.0 + index,
            "relHum": 0.50 + index * 0.001,
            "winSpe": 2.0 + index,
            "HDirNor": 100.0 + index,
            "PriceElectricPowerConstant": np.full(count, 0.10),
            "PriceElectricPowerDynamic": 0.10 + index * 0.01,
            "PriceElectricPowerHighlyDynamic": 0.10 + index * 0.02,
        }
        payload = {point: data[point].tolist() for point in point_names}
        return ForecastResponse(status=200, message="", payload=payload)

    def results(
        self,
        point_names: list[str],
        start_time_seconds: int,
        final_time_seconds: int,
    ) -> ResultsResponse:
        self.results_calls.append(
            {
                "point_names": point_names,
                "start_time_seconds": start_time_seconds,
                "final_time_seconds": final_time_seconds,
            }
        )
        count = max(1, (final_time_seconds - start_time_seconds) // self.step_seconds + 1)
        index = np.arange(count, dtype=np.float64)
        data = {
            "time": np.arange(count, dtype=np.float64) * self.step_seconds
            + start_time_seconds,
            "zon_reaTRooAir_y": 273.15 + 20.0 + index,
            "fcu_reaPCoo_y": 100.0 + index,
            "fcu_reaPFan_y": 10.0 + index,
            "fcu_reaPHea_y": 20.0 + index,
            "TDryBul": 273.15 + 5.0 + index,
            "relHum": 0.40 + index * 0.01,
            "winSpe": 1.0 + index,
            "HDirNor": 50.0 + index,
        }
        payload = {point: data[point].tolist() for point in point_names}
        return ResultsResponse(status=200, message="", payload=payload)

    def get_kpis(self) -> KpiResponse:
        return KpiResponse(
            status=200,
            message="",
            payload=KpiPayload(cost_tot=1.0, tdis_tot=2.0),
        )

    def stop(self) -> StopResponse:
        self.stop_calls += 1
        return StopResponse(
            status=200,
            message="",
            payload=StopPayload(stopped=True),
        )

    def _payload(self) -> dict[str, float]:
        advance_index = len(self.advance_calls)
        return {
            "time": float(self.current_time_seconds),
            "zon_reaTRooAir_y": 273.15 + 20.0 + advance_index,
            "fcu_reaPCoo_y": 100.0,
            "fcu_reaPFan_y": 10.0,
            "fcu_reaPHea_y": 20.0,
        }
