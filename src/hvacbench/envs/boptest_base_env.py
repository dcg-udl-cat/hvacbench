from abc import ABC
from typing import Optional

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from hvacbench.boptest.base import BaseBoptestClient, ResultsResponse
from hvacbench.boptest.bestest_air import BestestAir
from hvacbench.boptest.client import BoptestClient
from hvacbench.boptest.testcase import BoptestTestcase
from hvacbench.config import EnvConfig
from hvacbench.energy_price import EnergyPriceType
from hvacbench.envs.base import BaseEnv
from hvacbench.rewards.base import RewardStrategy
from hvacbench.schemas import FloatArray, Observation


SECONDS_PER_DAY = 24 * 60 * 60


class BaseBoptestEnv(BaseEnv, ABC):
    """Shared mechanics for BOPTEST-backed environments."""

    _WARMUP_PERIOD_SECONDS = 0

    def __init__(
        self,
        reward: RewardStrategy,
        config: EnvConfig,
        energy_price_type: EnergyPriceType = EnergyPriceType.DYNAMIC,
        testcase: Optional[BoptestTestcase] = None,
        main_client: Optional[BaseBoptestClient] = None,
        start_day: int = 0,
    ) -> None:
        self.config = config
        self.reward = reward
        self.testcase = testcase or BestestAir(energy_price_type=energy_price_type)
        self.start_day = int(start_day)
        if self.start_day < 0:
            raise ValueError("start_day must be greater than or equal to 0.")
        self.start_time_seconds = self.start_day * SECONDS_PER_DAY

        self.main_client = main_client or BoptestClient(self.testcase.base_url)

        self.current_time_seconds = self.start_time_seconds
        self.elapsed_episode_seconds = 0

        self.weather_history: FloatArray = np.zeros(
            (self.config.history_length, self.config.n_weather),
            dtype=np.float64,
        )
        self.control_history: FloatArray = np.zeros(
            (self.config.history_length, self.config.n_controls),
            dtype=np.float64,
        )
        self.state_history: FloatArray = np.zeros(
            (self.config.history_length, self.config.n_states),
            dtype=np.float64,
        )

    def _reset_main_client_context(self) -> None:
        self.current_time_seconds = self.start_time_seconds
        self.elapsed_episode_seconds = 0

        self._initialize_client(self.main_client)
        initial_weather_history = self._get_initial_weather_history_forecast()
        self._advance_initial_context(self.main_client)

        self.current_time_seconds = (
            self.start_time_seconds
            + self.config.history_length * self.testcase.step_period_seconds
        )
        self._initialize_histories_from_results(initial_weather_history)

    def _initialize_client(self, client: BaseBoptestClient) -> None:
        client.set_step(self.testcase.step_period_seconds)
        client.set_scenario(self.testcase.energy_price_type)
        client.initialize(
            start_time_seconds=self.start_time_seconds,
            warmup_period_seconds=self._WARMUP_PERIOD_SECONDS,
        )

    def _advance_initial_context(self, client: BaseBoptestClient) -> None:
        default_inputs = client.control_row_to_inputs(
            self.testcase,
            self.testcase.default_control_row(),
        )
        for _ in range(self.config.history_length):
            client.advance(default_inputs)

    def get_obs(self) -> Observation:
        return Observation(
            weather_history=self.weather_history.copy(),
            control_history=self.control_history.copy(),
            state_history=self.state_history.copy(),
            weather_forecast=self._get_weather_forecast(),
            energy_price_forecast=self._get_energy_price_forecast(),
        )

    @jaxtyped(typechecker=beartype)
    def get_random_control_plan(
        self,
    ) -> Float[np.ndarray, "{self.config.horizon} {self.config.n_controls}"]:
        control_plan = np.zeros(
            (self.config.horizon, self.config.n_controls),
            dtype=np.float64,
        )
        rng = np.random.default_rng()
        control_plan[:, 0] = rng.uniform(18.0, 22.0, size=self.config.horizon)
        control_plan[:, 1] = rng.uniform(22.0, 26.0, size=self.config.horizon)
        return control_plan

    @jaxtyped(typechecker=beartype)
    def _get_weather_forecast(
        self,
    ) -> Float[np.ndarray, "{self.config.horizon} {self.config.n_weather}"]:
        return self._get_weather_forecast_steps(self.config.horizon)

    @jaxtyped(typechecker=beartype)
    def _get_initial_weather_history_forecast(
        self,
    ) -> Float[np.ndarray, "{self.config.history_length} {self.config.n_weather}"]:
        return self._get_weather_forecast_steps(self.config.history_length)

    def _get_weather_forecast_steps(self, steps: int) -> FloatArray:
        response = self.main_client.forecast(
            point_names=self.testcase.weather_forecast_points(),
            horizon_seconds=self._forecast_horizon_seconds(steps),
            interval_seconds=self.testcase.step_period_seconds,
        )

        columns: list[np.ndarray] = []
        for point_name in self.testcase.weather_forecast_points():
            values = self._series_from_payload(response.series, point_name)
            columns.append(
                self.main_client.convert_weather_point(
                    self.testcase,
                    point_name,
                    values,
                )
            )

        return np.column_stack(columns).astype(np.float64)

    @jaxtyped(typechecker=beartype)
    def _get_energy_price_forecast(
        self,
    ) -> Float[np.ndarray, "{self.config.horizon}"]:
        price_point = self.testcase.energy_price_forecast_point()
        response = self.main_client.forecast(
            point_names=[price_point],
            horizon_seconds=self._forecast_horizon_seconds(self.config.horizon),
            interval_seconds=self.testcase.step_period_seconds,
        )
        values = self._series_from_payload(response.series, price_point)
        return values.astype(np.float64)

    def _forecast_horizon_seconds(self, steps: int) -> int:
        return (steps - 1) * self.testcase.step_period_seconds

    @staticmethod
    def _series_from_payload(
        payload: dict[str, list[float]],
        point_name: str,
    ) -> FloatArray:
        if point_name not in payload:
            raise KeyError(f"BOPTEST payload missing point {point_name!r}.")
        return np.asarray(payload[point_name], dtype=np.float64)

    def _initialize_histories_from_results(
        self,
        initial_weather_history: FloatArray,
    ) -> None:
        start_time = self.current_time_seconds - (
            self.config.history_length * self.testcase.step_period_seconds
        )
        response = self.main_client.results(
            point_names=self.testcase.result_points(),
            start_time_seconds=start_time,
            final_time_seconds=self.current_time_seconds,
        )

        state_rows = self._state_rows_from_results(response)
        if state_rows.size == 0 or initial_weather_history.size == 0:
            raise ValueError("BOPTEST results did not include history rows.")

        self.state_history = self._left_pad_rows(state_rows, self.config.history_length)
        self.weather_history = self._left_pad_rows(
            initial_weather_history,
            self.config.history_length,
        )
        self.control_history = np.tile(
            self.testcase.default_control_row(),
            (self.config.history_length, 1),
        )

    def _state_rows_from_results(self, response: ResultsResponse) -> FloatArray:
        rows = self.main_client.state_rows_from_results(self.testcase, response)
        return rows[-self.config.history_length :]

    @staticmethod
    def _left_pad_rows(rows: FloatArray, target_rows: int) -> FloatArray:
        if rows.shape[0] >= target_rows:
            return rows[-target_rows:].astype(np.float64)

        pad_count = target_rows - rows.shape[0]
        padding = np.repeat(rows[0:1], pad_count, axis=0)
        return np.concatenate([padding, rows], axis=0).astype(np.float64)

    def _append_to_histories(
        self,
        next_weather: FloatArray,
        next_control: FloatArray,
        next_state: FloatArray,
    ) -> None:
        self.weather_history = np.concatenate(
            [self.weather_history[1:], next_weather.reshape(1, -1)],
            axis=0,
        )
        self.control_history = np.concatenate(
            [self.control_history[1:], next_control.reshape(1, -1)],
            axis=0,
        )
        self.state_history = np.concatenate(
            [self.state_history[1:], next_state.reshape(1, -1)],
            axis=0,
        )

    def _control_row_to_boptest_inputs(self, control_row: FloatArray) -> dict[str, float]:
        if control_row.shape != (self.config.n_controls,):
            msg = (
                "Invalid control row shape for bestest_air: "
                f"expected ({self.config.n_controls},), got {control_row.shape}."
            )
            raise ValueError(msg)
        return self.main_client.control_row_to_inputs(self.testcase, control_row)

    def _extract_state_from_payload(self, payload: dict[str, float]) -> FloatArray:
        return self.main_client.extract_state_from_values(self.testcase, payload)

    def close(self) -> None:
        self.main_client.stop()

    def __enter__(self) -> "BaseBoptestEnv":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = ["BaseBoptestEnv"]
