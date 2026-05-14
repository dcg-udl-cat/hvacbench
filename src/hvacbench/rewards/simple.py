from typing import Any
import numpy as np

from hvacbench.schemas import FloatArray, Observation
from hvacbench.rewards.base import RewardStrategy
from hvacbench.config import EnvConfig

class SimpleReward(RewardStrategy):
    """A horizon reward that penalizes comfort violations and energy cost."""
    
    def __init__(
        self,
        config: EnvConfig,
        comfort_weight: float = 1.0,
        energy_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        temp_min: float = 20.0,
        temp_max: float = 24.0,
    ):
        self.config = config
        self.comfort_weight = comfort_weight
        self.energy_weight = energy_weight
        self.smoothness_weight = smoothness_weight
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.timestep_hours = config.timestep_minutes / 60.0

    def compute_reward(
        self,
        predicted_states: FloatArray,
        control_plan: FloatArray,
        weather_forecast: FloatArray,
        energy_price_forecast: FloatArray,
        current_obs: Observation,
        info: dict[str, Any],
    ) -> float:
        # Assuming state format: 0 -> Room Temp, 1 -> HVAC Power (W)
        room_temp = predicted_states[:, 0]
        hvac_power_w = predicted_states[:, 1]
        
        # Comfort penalty (deviation from bounds over the horizon)
        too_cold = np.maximum(0, self.temp_min - room_temp)
        too_hot = np.maximum(0, room_temp - self.temp_max)
        comfort_penalty = np.sum(too_cold + too_hot)
        
        # Energy cost
        power_kw = hvac_power_w / 1000.0
        energy_kwh = power_kw * self.timestep_hours
        energy_cost = np.sum(energy_kwh * energy_price_forecast)
        
        # Smoothness penalty
        diffs = np.diff(control_plan, axis=0)
        smoothness_penalty = float(np.sum(np.abs(diffs)))
        
        reward = -(
            self.comfort_weight * comfort_penalty +
            self.energy_weight * energy_cost +
            self.smoothness_weight * smoothness_penalty
        )
        
        info["reward_components"] = {
            "comfort_penalty": float(comfort_penalty),
            "energy_cost": float(energy_cost),
            "smoothness_penalty": float(smoothness_penalty),
        }
        
        return float(reward)
