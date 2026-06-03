from abc import ABC, abstractmethod
import numpy as np
from jaxtyping import Float

class BaseTTM(ABC):
    @abstractmethod
    def predict(
        self,
        weather_history: Float[np.ndarray, "history_length n_weather"],
        control_history: Float[np.ndarray, "history_length n_controls"],
        state_history: Float[np.ndarray, "history_length n_states"],
        weather_forecast: Float[np.ndarray, "horizon n_weather"],
        control_plan: Float[np.ndarray, "horizon n_controls"],
    ) -> Float[np.ndarray, "horizon n_states"]: ...
