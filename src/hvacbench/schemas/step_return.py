from typing import Any, Tuple
from .observation import Observation

# A Gymnasium-like step return type: next_obs, reward, terminated, truncated, info
StepReturn = Tuple[Observation, float, bool, bool, dict[str, Any]]
