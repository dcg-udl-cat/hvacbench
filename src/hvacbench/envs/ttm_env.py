import numpy as np
from typing import Any, Tuple

from hvacbench.schemas import FloatArray, Observation, StepReturn
from hvacbench.config import EnvConfig
from hvacbench.providers.base import BaseProvider
from hvacbench.rewards.base import RewardStrategy
from hvacbench.models.base import BuildingModel
from hvacbench.envs.base import BaseEnv
from hvacbench.utils.validation import validate_shape

class TTMEnv(BaseEnv):
    """Environment backed by a forecasting model / digital twin."""

    def __init__(
        self,
        config: EnvConfig,
        provider: BaseProvider,
        reward: RewardStrategy,
        model: BuildingModel,
    ):
        self.config = config
        self.provider = provider
        self.reward = reward
        self.model = model
        
        self.t = 0
        self.weather_history: FloatArray = np.zeros((0,0))
        self.control_history: FloatArray = np.zeros((0,0))
        self.state_history: FloatArray = np.zeros((0,0))

    def reset(self) -> Tuple[Observation, dict[str, Any]]:
        self.t = 0
        hl = self.config.history_length
        self.weather_history = self.provider.get_initial_weather_history(hl)
        self.control_history = self.provider.get_initial_control_history(hl)
        self.state_history = self.provider.get_initial_state_history(hl)
        
        validate_shape(self.weather_history, (hl, self.config.n_weather), "weather_history")
        validate_shape(self.control_history, (hl, self.config.n_controls), "control_history")
        validate_shape(self.state_history, (hl, self.config.n_states), "state_history")

        return self.get_obs(), {}

    def get_obs(self) -> Observation:
        hz = self.config.horizon
        weather_forecast = self.provider.get_weather_forecast(self.t, hz)
        energy_price_forecast = self.provider.get_energy_price_forecast(self.t, hz)
        
        return Observation(
            weather_history=self.weather_history.copy(),
            control_history=self.control_history.copy(),
            state_history=self.state_history.copy(),
            weather_forecast=weather_forecast,
            energy_price_forecast=energy_price_forecast,
        )

    def _predict_next_states(self, obs: Observation, control_plan: FloatArray) -> FloatArray:
        predicted_states = self.model.predict(
            weather_history=obs.weather_history,
            control_history=obs.control_history,
            state_history=obs.state_history,
            weather_forecast=obs.weather_forecast,
            control_plan=control_plan,
        )
        validate_shape(predicted_states, (self.config.horizon, self.config.n_states), "predicted_states")
        return predicted_states

    def _update_histories(self, obs: Observation, control_plan: FloatArray, predicted_states: FloatArray) -> None:
        applied_control = control_plan[0:1]
        next_state = predicted_states[0:1]
        next_weather = obs.weather_forecast[0:1]
        
        self.control_history = np.concatenate([self.control_history[1:], applied_control], axis=0)
        self.state_history = np.concatenate([self.state_history[1:], next_state], axis=0)
        self.weather_history = np.concatenate([self.weather_history[1:], next_weather], axis=0)

    def step(self, control_plan: FloatArray) -> StepReturn:
        validate_shape(control_plan, (self.config.horizon, self.config.n_controls), "action (control_plan)")

        obs = self.get_obs()
        predicted_states = self._predict_next_states(obs, control_plan)
        
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
        self.t += 1
        
        terminated = False
        truncated = self.t >= self.provider.total_timesteps - self.config.horizon
        
        return self.get_obs(), reward_val, terminated, truncated, info
