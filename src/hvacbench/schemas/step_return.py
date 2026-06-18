from typing import Any
from .observation import Observation

# A Gymnasium-like step return type: next_obs, reward, terminated, truncated, info
StepReturn = tuple[Observation, float, bool, bool, dict[str, Any]]
