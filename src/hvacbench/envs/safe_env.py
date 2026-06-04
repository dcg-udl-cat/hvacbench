from typing import Any, Tuple, cast

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from hvacbench.schemas import FloatArray, Observation, StepReturn
from hvacbench.envs.base import BaseEnv
from hvacbench.safety.control_safety import ControlSafetyFilter

class SafeEnv(BaseEnv):
    """Decorator around another Env that applies a safety layer to proposed controls."""

    def __init__(self, env: BaseEnv, safety_filter: ControlSafetyFilter):
        self.env = env
        self.config = cast(Any, env).config
        self.safety_filter = safety_filter

    def get_obs(self) -> Observation:
        return self.env.get_obs()

    def reset(self) -> Tuple[Observation, dict[str, Any]]:
        return self.env.reset()

    @jaxtyped(typechecker=beartype)
    def get_random_control_plan(
        self,
    ) -> Float[np.ndarray, "{self.config.horizon} {self.config.n_controls}"]:
        return self.safety_filter.apply(self.env.get_random_control_plan())

    def step(self, action: FloatArray) -> StepReturn:
        # Retrieve previous control conceptually from env if available via obs. 
        # For simplicity, safe_control_plan applies stateless rules here based on the filter logic.
        safe_action = self.safety_filter.apply(action)
        
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        
        info["original_action"] = action
        info["safe_action"] = safe_action
        
        return obs, reward, terminated, truncated, info
