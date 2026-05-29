import numpy as np
from hvacbench.schemas import FloatArray
from hvacbench.config import EnvConfig
from hvacbench.providers.base import BaseProvider

class MockProvider(BaseProvider):
    """Provides deterministic/pseudo-random weather/energy forecasts for testing."""
    
    def __init__(self, config: EnvConfig, total_timesteps: int = 10000, seed: int = 42):
        self.config = config
        self._total_timesteps = total_timesteps
        self.rng = np.random.default_rng(seed)

    @property
    def total_timesteps(self) -> int:
        return self._total_timesteps

    def get_weather_forecast(self, t: int, horizon: int) -> FloatArray:
        # Simplistic mock: sinusoidal temperature + random noise
        times = np.arange(t, t + horizon)
        temp = 15.0 + 10.0 * np.sin(2 * np.pi * times / 96) # Mock daily cycle
        humidity = np.clip(50 + 10 * np.sin(2 * np.pi * times / 96 + np.pi/2), 0, 100)
        wind = np.clip(self.rng.normal(3.0, 1.0, size=horizon), 0, 20)
        solar = np.clip(400 * np.sin(2 * np.pi * times / 96), 0, 1000)
        
        return np.stack([temp, humidity, wind, solar], axis=1)

    def get_energy_price_forecast(self, t: int, horizon: int) -> FloatArray:
        # Simplistic price mock
        times = np.arange(t, t + horizon)
        price = 0.10 + 0.05 * np.sin(2 * np.pi * times / 96)
        return price

    def get_initial_weather_history(self, history_length: int) -> FloatArray:
        return self.get_weather_forecast(-history_length, history_length)

    def get_initial_control_history(self, history_length: int) -> FloatArray:
        controls = np.zeros((history_length, self.config.n_controls))
        controls[:, 0] = 20.0 # Heating
        controls[:, 1] = 24.0 # Cooling
        return controls

    def get_initial_state_history(self, history_length: int) -> FloatArray:
        states = np.zeros((history_length, self.config.n_states))
        states[:, 0] = 22.0 # Temperature
        states[:, 1] = 500.0 # Power W
        return states

    def get_random_action(self) -> FloatArray:
        """Sample a random control trajectory within plausible setpoints."""
        action = np.zeros((self.config.horizon, self.config.n_controls))
        action[:, 0] = self.rng.uniform(18.0, 22.0, size=self.config.horizon)  # Heating
        action[:, 1] = self.rng.uniform(22.0, 26.0, size=self.config.horizon)  # Cooling
        return action
