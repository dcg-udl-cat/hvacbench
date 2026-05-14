from abc import ABC, abstractmethod
from hvacbench.schemas import FloatArray

class BuildingModel(ABC):
    @abstractmethod
    def predict(
        self,
        weather_history: FloatArray,
        control_history: FloatArray,
        state_history: FloatArray,
        weather_forecast: FloatArray,
        control_plan: FloatArray,
    ) -> FloatArray: ...