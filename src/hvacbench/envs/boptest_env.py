from typing import Any, Tuple
from hvacbench.schemas import FloatArray, Observation, StepReturn
from hvacbench.envs.base import BaseEnv

class BoptestEnv(BaseEnv):
    """Stub class for a building simulator environment via BOPTEST."""
    
    def get_obs(self) -> Observation:
        raise NotImplementedError("BoptestEnv.get_obs() is not implemented yet.")
        
    def step(self, action: FloatArray) -> StepReturn:
        raise NotImplementedError("BoptestEnv.step() is not implemented yet.")
        
    def reset(self) -> Tuple[Observation, dict[str, Any]]:
        raise NotImplementedError("BoptestEnv.reset() is not implemented yet.")
