from typing import Any, Self

import requests

from hvacbench.boptest.base import (
    AdvanceResponse,
    BaseBoptestClient,
    BoptestResponse,
    ForecastPoint,
    ForecastResponse,
    GetForecastPointsResponse,
    GetInputsResponse,
    GetMeasurementsResponse,
    InitializeResponse,
    Input,
    KpiPayload,
    KpiResponse,
    Measurement,
    ResultsResponse,
    ScenarioPayload,
    SelectResponse,
    SetScenarioResponse,
    SetStepResponse,
    StopPayload,
    StopResponse,
)


class _RawBoptestResponse(BoptestResponse):
    payload: object | None = None


class BoptestClient(BaseBoptestClient):
    """Small bestest_air-only wrapper around the BOPTEST HTTP API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.testcase = "bestest_air"
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.testid = self._select_testcase()
        self._stopped = False

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint}/{self.testid}"

    def _request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> _RawBoptestResponse:
        response = self._send_request(method, url, payload)
        envelope = self._decode_response(response, method, url)
        self._raise_for_boptest_error(envelope, method, url)
        return envelope

    def _send_request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
    ) -> requests.Response:
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"BOPTEST request failed: {method} {url}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"BOPTEST HTTP error: {response.status_code} for {method} {url}"
            )
        return response

    @staticmethod
    def _decode_response(
        response: requests.Response,
        method: str,
        url: str,
    ) -> _RawBoptestResponse:
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"BOPTEST returned non-JSON response for {method} {url}"
            ) from exc

        if not isinstance(body, dict):
            raise RuntimeError(f"BOPTEST response is not an object: {body!r}")

        if "status" in body and "message" in body and "payload" in body:
            return _RawBoptestResponse.model_validate(body)

        # /testcases/{testcase}/select commonly returns {"testid": "..."}.
        return _RawBoptestResponse(status=200, message="", payload=body)

    @staticmethod
    def _raise_for_boptest_error(
        response: BoptestResponse,
        method: str,
        url: str,
    ) -> None:
        if response.status != 200:
            raise RuntimeError(
                f"BOPTEST API error for {method} {url}: "
                f"status={response.status}, message={response.message!r}"
            )

    def _select_testcase(self) -> str:
        response = self._request(
            "POST",
            f"{self.base_url}/testcases/bestest_air/select",
        )
        select_response = SelectResponse(
            status=response.status,
            message=response.message,
            payload={"testid": str(self._payload_dict(response)["testid"])},
        )
        return select_response.testid

    @staticmethod
    def _payload_dict(response: _RawBoptestResponse) -> dict[str, Any]:
        if not isinstance(response.payload, dict):
            raise RuntimeError(f"BOPTEST payload is not an object: {response.payload!r}")
        return response.payload

    def get_inputs(self) -> GetInputsResponse:
        response = self._request("GET", self._url("inputs"))
        payload = self._payload_dict(response)
        return GetInputsResponse(
            status=response.status,
            message=response.message,
            payload={
                name: self._input_from_metadata(name, metadata)
                for name, metadata in payload.items()
            },
        )

    def get_measurements(self) -> GetMeasurementsResponse:
        response = self._request("GET", self._url("measurements"))
        payload = self._payload_dict(response)
        return GetMeasurementsResponse(
            status=response.status,
            message=response.message,
            payload={
                name: self._measurement_from_metadata(name, metadata)
                for name, metadata in payload.items()
            },
        )

    def get_forecast_points(self) -> GetForecastPointsResponse:
        response = self._request("GET", self._url("forecast_points"))
        payload = self._payload_dict(response)
        return GetForecastPointsResponse(
            status=response.status,
            message=response.message,
            payload={
                name: self._forecast_point_from_metadata(name, metadata)
                for name, metadata in payload.items()
            },
        )

    def set_step(self, step_seconds: int) -> SetStepResponse:
        response = self._request(
            "PUT",
            self._url("step"),
            {"step": int(step_seconds)},
        )
        return SetStepResponse(
            status=response.status,
            message=response.message,
            payload=self._step_payload(response),
        )

    def set_scenario(self, electricity_price_type: str) -> SetScenarioResponse:
        response = self._request(
            "PUT",
            self._url("scenario"),
            {"electricity_price": electricity_price_type},
        )
        payload = self._payload_dict(response)
        return SetScenarioResponse(
            status=response.status,
            message=response.message,
            payload=ScenarioPayload.model_validate(payload),
        )

    def initialize(
        self,
        start_time_seconds: int,
        warmup_period_seconds: int,
    ) -> InitializeResponse:
        response = self._request(
            "PUT",
            self._url("initialize"),
            {
                "start_time": int(start_time_seconds),
                "warmup_period": int(warmup_period_seconds),
            },
        )
        return InitializeResponse(
            status=response.status,
            message=response.message,
            payload=self._float_payload(response),
        )

    def advance(self, inputs: dict[str, float]) -> AdvanceResponse:
        response = self._request("POST", self._url("advance"), inputs)
        return AdvanceResponse(
            status=response.status,
            message=response.message,
            payload=self._float_payload(response),
        )

    def forecast(
        self,
        point_names: list[str],
        horizon_seconds: int,
        interval_seconds: int,
    ) -> ForecastResponse:
        response = self._request(
            "PUT",
            self._url("forecast"),
            {
                "point_names": point_names,
                "horizon": int(horizon_seconds),
                "interval": int(interval_seconds),
            },
        )
        return ForecastResponse(
            status=response.status,
            message=response.message,
            payload=self._series_payload(response),
        )

    def results(
        self,
        point_names: list[str],
        start_time_seconds: int,
        final_time_seconds: int,
    ) -> ResultsResponse:
        response = self._request(
            "PUT",
            self._url("results"),
            {
                "point_names": point_names,
                "start_time": int(start_time_seconds),
                "final_time": int(final_time_seconds),
            },
        )
        return ResultsResponse(
            status=response.status,
            message=response.message,
            payload=self._series_payload(response),
        )

    def get_kpis(self) -> KpiResponse:
        response = self._request("GET", self._url("kpi"))
        payload = self._payload_dict(response)
        return KpiResponse(
            status=response.status,
            message=response.message,
            payload=KpiPayload.model_validate(payload),
        )

    def stop(self) -> StopResponse:
        response = self._send_request(
            "PUT",
            f"{self.base_url}/stop/{self.testid}",
            payload=None,
        )
        self._stopped = True
        return StopResponse(
            status=response.status_code,
            message=response.text,
            payload=StopPayload(stopped=True),
        )

    @staticmethod
    def _input_from_metadata(name: str, metadata: Any) -> Input:
        data = metadata if isinstance(metadata, dict) else {}
        return Input(
            name=name,
            description=data.get("Description"),
            unit=data.get("Unit"),
            minimum=data.get("Minimum"),
            maximum=data.get("Maximum"),
        )

    @staticmethod
    def _measurement_from_metadata(name: str, metadata: Any) -> Measurement:
        data = metadata if isinstance(metadata, dict) else {}
        return Measurement(
            name=name,
            description=data.get("Description"),
            unit=data.get("Unit"),
            minimum=data.get("Minimum"),
            maximum=data.get("Maximum"),
        )

    @staticmethod
    def _forecast_point_from_metadata(name: str, metadata: Any) -> ForecastPoint:
        data = metadata if isinstance(metadata, dict) else {}
        return ForecastPoint(
            name=name,
            description=data.get("Description"),
            unit=data.get("Unit"),
        )

    @classmethod
    def _step_payload(cls, response: _RawBoptestResponse) -> float | str:
        if isinstance(response.payload, (float, int, str)):
            return response.payload
        if isinstance(response.payload, dict):
            step = response.payload.get("step")
            if isinstance(step, (float, int, str)):
                return step
        raise RuntimeError(f"BOPTEST step payload is not a scalar: {response.payload!r}")

    @classmethod
    def _float_payload(cls, response: _RawBoptestResponse) -> dict[str, float]:
        payload = cls._payload_dict(response)
        return {key: float(value) for key, value in payload.items()}

    @classmethod
    def _series_payload(cls, response: _RawBoptestResponse) -> dict[str, list[float]]:
        payload = cls._payload_dict(response)
        return {
            key: [float(value) for value in values]
            for key, values in payload.items()
            if isinstance(values, list)
        }

    def close(self) -> None:
        if not self._stopped:
            try:
                self.stop()
            finally:
                self.session.close()
        else:
            self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
