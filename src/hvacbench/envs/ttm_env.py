import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped
from typing import Any, Tuple

from hvacbench.schemas import FloatArray, Observation, StepReturn
from hvacbench.config import EnvConfig, TTMVariables
from hvacbench.providers.base import BaseProvider
from hvacbench.rewards.base import RewardStrategy
from hvacbench.models.base import BaseTTM
from hvacbench.envs.base import BaseEnv


class TTMEnv(BaseEnv):
    """Environment backed by a forecasting model / digital twin."""

    def __init__(
        self,
        config: EnvConfig,
        provider: BaseProvider,
        reward: RewardStrategy,
        model: BaseTTM,
        variables: TTMVariables = TTMVariables(),
    ):
        self.config = config
        self.provider = provider
        self.reward = reward
        self.model = model
        self.variables = variables
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

    def reset(self) -> Tuple[Observation, dict[str, Any]]:
        self.current_timestep = 0
        hl = self.history_buffer_length
        self.weather_history = self.provider.get_initial_weather_history(hl)
        self.control_history = self.provider.get_initial_control_history(hl)
        self.state_history = self.provider.get_initial_state_history(hl)

        return self.get_obs(), {}

    def get_obs(self) -> Observation:
        hz = self.config.horizon
        weather_forecast = self.provider.get_weather_forecast(self.current_timestep, hz)
        energy_price_forecast = self.provider.get_energy_price_forecast(
            self.current_timestep, hz
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
            self.current_timestep,
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
