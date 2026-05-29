from abc import ABC, abstractmethod
from hvacbench.schemas import FloatArray

class BaseProvider(ABC):
    @abstractmethod
    def get_weather_forecast(self, t: int, horizon: int) -> FloatArray: ...
    @abstractmethod
    def get_energy_price_forecast(self, t: int, horizon: int) -> FloatArray: ...
    
    @abstractmethod
    def get_initial_weather_history(self, history_length: int) -> FloatArray: ...
    @abstractmethod
    def get_initial_control_history(self, history_length: int) -> FloatArray: ...
    @abstractmethod
    def get_initial_state_history(self, history_length: int) -> FloatArray: ...
    
    @property
    @abstractmethod
    def total_timesteps(self) -> int: ...

    @abstractmethod
    def get_random_action(self) -> FloatArray:
        """Sample a random action for the environment."""
        ...
