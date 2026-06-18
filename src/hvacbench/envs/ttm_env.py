from os import PathLike
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from hvacbench.energy_price import EnergyPriceType
from hvacbench.models import TTM
from hvacbench.providers import BestestAirCsvProvider
from hvacbench.schemas import FloatArray, Observation, StepReturn
from hvacbench.config import EnvConfig, TTMVariables
from hvacbench.providers.base import BaseProvider
from hvacbench.rewards.base import RewardStrategy
from hvacbench.models.base import BaseTTM
from hvacbench.envs.base import BaseEnv


SECONDS_PER_DAY = 24 * 60 * 60


class TTMEnv(BaseEnv):
    """Environment backed by a forecasting model / digital twin."""

    def __init__(
        self,
        config: EnvConfig,
        reward: RewardStrategy,
        model_path: str | PathLike[str] | None = None,
        energy_price_type: EnergyPriceType = EnergyPriceType.DYNAMIC,
        variables: TTMVariables = TTMVariables(),
        model: BaseTTM | None = None,
        provider: BaseProvider | None = None,
        building_data_path: str | PathLike[str] | None = None,
        electricity_price_data_path: str | PathLike[str] | None = None,
        start_day: int = 0,
    ) -> None:
        self.config = config
        self.reward = reward
        self.variables = variables
        self.model = model or self._load_model(model_path)
        self.provider = provider or BestestAirCsvProvider(
            config=config,
            energy_price_type=energy_price_type,
            variables=variables,
            building_data_path=building_data_path,
            electricity_price_data_path=electricity_price_data_path,
        )
        self.start_day = int(start_day)
        if self.start_day < 0:
            raise ValueError("start_day must be greater than or equal to 0.")
        self.start_timestep = self._start_timestep_from_day(self.start_day)
        self._validate_variables()
        self.model_prediction_length = int(self.model.prediction_length)
        self._validate_prediction_length()
        self.model_context_length = self.model.context_length
        self.history_buffer_length = max(
            self.config.history_length, self.model_context_length
        )

        self.current_timestep = 0

        self.weather_history: FloatArray = np.zeros(
            (self.history_buffer_length, self.config.n_weather),
            dtype=np.float64,
        )
        self.control_history: FloatArray = np.zeros(
            (self.history_buffer_length, self.config.n_controls),
            dtype=np.float64,
        )
        self.state_history: FloatArray = np.zeros(
            (self.history_buffer_length, self.config.n_states),
            dtype=np.float64,
        )

        self.reset()

    def _load_model(self, model_path: str | PathLike[str] | None) -> BaseTTM:
        if model_path is None:
            raise ValueError("TTMEnv requires either model or model_path.")
        return TTM(
            config=self.config,
            model_path=str(model_path),
            variables=self.variables,
        )

    def _start_timestep_from_day(self, start_day: int) -> int:
        start_seconds = start_day * SECONDS_PER_DAY
        return start_seconds // self.config.step_period_seconds

    def _provider_timestep(self) -> int:
        return self.start_timestep + self.current_timestep

    def _validate_variables(self) -> None:
        if len(self.variables.state_names) != self.config.n_states:
            raise ValueError("TTMVariables.state_vars must define exactly two states.")
        if len(self.variables.weather_names) != self.config.n_weather:
            raise ValueError(
                "TTMVariables.weather_vars must define exactly four weather variables."
            )
        if len(self.variables.control_names) != self.config.n_controls:
            raise ValueError(
                "TTMVariables.control_vars must define exactly two controls."
            )

    def _validate_prediction_length(self) -> None:
        if self.config.horizon > self.model_prediction_length:
            raise ValueError(
                "TTM env horizon must be smaller than or equal to the model "
                f"prediction length: horizon={self.config.horizon}, "
                f"prediction_length={self.model_prediction_length}."
            )

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.current_timestep = 0
        hl = self.history_buffer_length
        self.weather_history = self.provider.get_initial_weather_history(
            hl,
            start_timestep=self.start_timestep,
        )
        self.control_history = self.provider.get_initial_control_history(
            hl,
            start_timestep=self.start_timestep,
        )
        self.state_history = self.provider.get_initial_state_history(
            hl,
            start_timestep=self.start_timestep,
        )

        return self.get_obs(), {}

    def get_obs(self) -> Observation:
        hz = self.config.horizon
        provider_timestep = self._provider_timestep()
        weather_forecast = self.provider.get_weather_forecast(provider_timestep, hz)
        energy_price_forecast = self.provider.get_energy_price_forecast(
            provider_timestep,
            hz,
        )

        return Observation(
            weather_history=self.weather_history[-self.config.history_length :].copy(),
            control_history=self.control_history[-self.config.history_length :].copy(),
            state_history=self.state_history[-self.config.history_length :].copy(),
            weather_forecast=weather_forecast,
            energy_price_forecast=energy_price_forecast,
        )

    @jaxtyped(typechecker=beartype)
    def get_random_control_plan(
        self,
    ) -> Float[np.ndarray, "{self.config.horizon} {self.config.n_controls}"]:
        return self.provider.get_random_action()

    @jaxtyped(typechecker=beartype)
    def _get_model_weather_forecast(
        self,
        weather_forecast: Float[
            np.ndarray,
            "{self.config.horizon} {self.config.n_weather}",
        ],
    ) -> Float[np.ndarray, "{self.model_prediction_length} {self.config.n_weather}"]:
        if self.model_prediction_length == self.config.horizon:
            return weather_forecast
        return self.provider.get_weather_forecast(
            self._provider_timestep(),
            self.model_prediction_length,
        )

    @jaxtyped(typechecker=beartype)
    def _get_model_control_plan(
        self,
        control_plan: Float[
            np.ndarray,
            "{self.config.horizon} {self.config.n_controls}",
        ],
    ) -> Float[np.ndarray, "{self.model_prediction_length} {self.config.n_controls}"]:
        if self.model_prediction_length == self.config.horizon:
            return control_plan

        replay_length = self.model_prediction_length - self.config.horizon
        replayed_tail = np.repeat(control_plan[-1:], replay_length, axis=0)
        return np.concatenate([control_plan, replayed_tail], axis=0)

    @jaxtyped(typechecker=beartype)
    def _predict_next_states(
        self, weather_forecast: FloatArray, control_plan: FloatArray
    ) -> Float[np.ndarray, "{self.config.horizon} {self.config.n_states}"]:
        model_weather_forecast = self._get_model_weather_forecast(weather_forecast)
        model_control_plan = self._get_model_control_plan(control_plan)
        predicted_states = self.model.predict(
            weather_history=self.weather_history[-self.model_context_length :].copy(),
            control_history=self.control_history[-self.model_context_length :].copy(),
            state_history=self.state_history[-self.model_context_length :].copy(),
            weather_forecast=model_weather_forecast,
            control_plan=model_control_plan,
        )
        return predicted_states[: self.config.horizon]

    def _update_histories(
        self, obs: Observation, control_plan: FloatArray, predicted_states: FloatArray
    ) -> None:
        applied_control = control_plan[0:1]
        next_state = predicted_states[0:1]
        next_weather = obs.weather_forecast[0:1]

        self.control_history = np.concatenate(
            [self.control_history[1:], applied_control], axis=0
        )
        self.state_history = np.concatenate(
            [self.state_history[1:], next_state], axis=0
        )
        self.weather_history = np.concatenate(
            [self.weather_history[1:], next_weather], axis=0
        )

    @jaxtyped(typechecker=beartype)
    def step(
        self,
        control_plan: Float[
            np.ndarray, "{self.config.horizon} {self.config.n_controls}"
        ],
    ) -> StepReturn:
        obs = self.get_obs()
        predicted_states = self._predict_next_states(obs.weather_forecast, control_plan)

        info: dict[str, Any] = {
            "predicted_states": predicted_states,
            "applied_control": control_plan[0].copy(),
            "control_plan": control_plan,
        }

        reward_val = self.reward.compute_reward(
            predicted_states=predicted_states,
            control_plan=control_plan,
            weather_forecast=obs.weather_forecast,
            energy_price_forecast=obs.energy_price_forecast,
            current_obs=obs,
            info=info,
        )

        self._update_histories(obs, control_plan, predicted_states)
        self.current_timestep += 1

        terminated = False
        elapsed_seconds = self.current_timestep * self.config.step_period_seconds
        truncated = elapsed_seconds >= self.config.total_simulation_seconds

        return self.get_obs(), reward_val, terminated, truncated, info
