import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped
from hvacbench.config import EnvConfig
from hvacbench.models.base import BaseTTM

class MockTTM(BaseTTM):
    """Produces plausible predicted states with correct shape for testing."""
    
    def __init__(
        self,
        config: EnvConfig,
        context_length: int | None = None,
        prediction_length: int | None = None,
    ):
        self.config = config
        self._context_length = context_length or config.history_length
        self._prediction_length = prediction_length or config.horizon

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def prediction_length(self) -> int:
        return self._prediction_length

    @jaxtyped(typechecker=beartype)
    def predict(
        self,
        weather_history: Float[np.ndarray, "{self.context_length} {self.config.n_weather}"],
        control_history: Float[np.ndarray, "{self.context_length} {self.config.n_controls}"],
        state_history: Float[np.ndarray, "{self.context_length} {self.config.n_states}"],
        weather_forecast: Float[np.ndarray, "{self.prediction_length} {self.config.n_weather}"],
        control_plan: Float[np.ndarray, "{self.prediction_length} {self.config.n_controls}"],
    ) -> Float[np.ndarray, "{self.prediction_length} {self.config.n_states}"]:
        hz = self.prediction_length

        predicted_states = np.zeros((hz, self.config.n_states))
        
        # Simple dynamics
        current_temp = state_history[-1, 0]
        
        for i in range(hz):
            heating_sp = control_plan[i, 0]
            cooling_sp = control_plan[i, 1]
            out_temp = weather_forecast[i, 0]
            
            target_temp = np.mean([heating_sp, cooling_sp])
            
            # Move towards target temp and outdoor temp
            current_temp = current_temp + 0.1 * (target_temp - current_temp) + 0.05 * (out_temp - current_temp)
            
            # Simple power heuristic
            power = 0.0
            if current_temp < heating_sp:
                power = (heating_sp - current_temp) * 1000
            elif current_temp > cooling_sp:
                power = (current_temp - cooling_sp) * 1000
                
            predicted_states[i, 0] = current_temp
            predicted_states[i, 1] = np.clip(power, 0, 5000)
            
        return predicted_states
