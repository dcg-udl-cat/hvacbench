from abc import ABC, abstractmethod
from typing import Tuple, Any

import numpy as np
from jaxtyping import Float

from hvacbench.schemas import FloatArray, Observation, StepReturn

class BaseEnv(ABC):
    """Abstract base class for all environments."""
    
    @abstractmethod
    def get_obs(self) -> Observation:
        pass
        
    @abstractmethod
    def step(self, action: FloatArray) -> StepReturn:
        pass
        
    @abstractmethod
    def reset(self) -> Tuple[Observation, dict[str, Any]]:
        pass

    @abstractmethod
    def get_random_control_plan(self) -> Float[np.ndarray, "horizon n_controls"]:
        pass
