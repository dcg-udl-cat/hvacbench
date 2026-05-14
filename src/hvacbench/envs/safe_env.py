from typing import Any, Tuple

from hvacbench.schemas import FloatArray, Observation, StepReturn
from hvacbench.envs.base import BaseEnv
from hvacbench.safety.control_safety import ControlSafetyFilter

class SafeEnv(BaseEnv):
    """Decorator around another Env that applies a safety layer to proposed controls."""

    def __init__(self, env: BaseEnv, safety_filter: ControlSafetyFilter):
        self.env = env
        self.safety_filter = safety_filter

    def get_obs(self) -> Observation:
        return self.env.get_obs()

    def reset(self) -> Tuple[Observation, dict[str, Any]]:
        return self.env.reset()

    def step(self, action: FloatArray) -> StepReturn:
        # Retrieve previous control conceptually from env if available via obs. 
        # For simplicity, safe_control_plan applies stateless rules here based on the filter logic.
        safe_action = self.safety_filter.apply(action)
        
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        
        info["original_action"] = action
        info["safe_action"] = safe_action
        
        return obs, reward, terminated, truncated, info
