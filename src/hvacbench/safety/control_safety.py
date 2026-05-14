import numpy as np
from hvacbench.schemas import FloatArray
from dataclasses import dataclass

@dataclass
class ControlSafetyFilter:
    """Clips controls to configured min/max limits and enforces logic."""
    heating_min: float = 15.0
    heating_max: float = 24.0
    cooling_min: float = 22.0
    cooling_max: float = 30.0

    def apply(self, control_plan: FloatArray) -> FloatArray:
        safe_plan = np.copy(control_plan)
        
        # Assuming index 0: Heating, 1: Cooling
        heating = safe_plan[:, 0]
        cooling = safe_plan[:, 1]
        
        heating = np.clip(heating, self.heating_min, self.heating_max)
        cooling = np.clip(cooling, self.cooling_min, self.cooling_max)
        
        # Enforce heating <= cooling
        # If heating > cooling, adjust cooling to max(cooling, heating)
        invalid_mask = heating > cooling
        cooling[invalid_mask] = heating[invalid_mask]
        
        # Ensure cooling still doesn't exceed its absolute max after adjustment
        # If it does, we must pull heating down too
        cap_mask = cooling > self.cooling_max
        cooling[cap_mask] = self.cooling_max
        heating[cap_mask] = self.cooling_max
        
        safe_plan[:, 0] = heating
        safe_plan[:, 1] = cooling
        
        return safe_plan
