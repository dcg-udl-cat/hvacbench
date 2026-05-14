from abc import ABC, abstractmethod
from typing import Any
from hvacbench.schemas import FloatArray, Observation

class RewardStrategy(ABC):
    @abstractmethod
    def compute_reward(
        self,
        predicted_states: FloatArray,
        control_plan: FloatArray,
        weather_forecast: FloatArray,
        energy_price_forecast: FloatArray,
        current_obs: Observation,
        info: dict[str, Any],
    ) -> float:
        pass