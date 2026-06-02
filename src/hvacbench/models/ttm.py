import logging
import numpy as np
import pandas as pd
from typing import Optional

from beartype import beartype
from jaxtyping import Float, jaxtyped

from hvacbench.config import EnvConfig
from hvacbench.models.base import BaseTTM

logger = logging.getLogger(__name__)


class TTM(BaseTTM):
    """Wrapper for TinyTimeMixer for single-step inference within the RL env."""

    def __init__(
            self,
            config: EnvConfig,
            model_path: str = "dummy",
            device: Optional[str] = None,
    ) -> None:
        """
        Initializes the model wrapper.

        Args:
            config (EnvConfig): Environment configuration (dims and names).
            model_path (str): Path to the saved model.
            device (Optional[str]): Device to run the model on ('cpu', 'cuda').
        """
        self.config = config
        self.context_length = config.history_length
        self.prediction_length = config.horizon

        self.state_vars = list(config.state_variables)
        self.weather_vars = list(config.weather_variables)
        self.control_vars = list(config.control_variables)

        import torch
        from tsfm_public import (
            TinyTimeMixerForPrediction,
            TimeSeriesPreprocessor,
            TimeSeriesForecastingPipeline,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyTimeMixerForPrediction.from_pretrained(model_path)
        self.tsp = TimeSeriesPreprocessor.from_pretrained(model_path)

        self._timestamps = self._build_timestamps()

        self._validate_model_config()

        self.pipeline = TimeSeriesForecastingPipeline(
            self.model,
            device=self.device,
            feature_extractor=self.tsp,
            batch_size=1,
            explode_forecasts=True,
        )

    def _validate_model_config(self) -> None:
        """Validate that model config matches expected dimensions."""
        if self.model.config.context_length != self.context_length:
            raise ValueError(
                f"Model context length ({self.model.config.context_length}) does not match config history length ({self.context_length})."
            )
        if self.model.config.prediction_length != self.prediction_length:
            raise ValueError(
                f"Model prediction length ({self.model.config.prediction_length}) does not match config horizon ({self.prediction_length})."
            )

        if sorted(self.tsp.target_columns) != sorted(self.state_vars):
            raise ValueError(
                f"Model target_columns ({self.tsp.target_columns}) does not match config state_variables ({self.state_vars})."
            )
        if sorted(self.tsp.observable_columns) != sorted(self.weather_vars):
            raise ValueError(
                f"Model observable_columns ({self.tsp.observable_columns}) does not match config weather_variables ({self.weather_vars})."
            )
        if sorted(self.tsp.control_columns) != sorted(self.control_vars):
            raise ValueError(
                f"Model control_columns ({self.tsp.control_columns}) does not match config control_variables ({self.control_vars})."
            )
        if self.tsp.freq != f"{self.config.timestep_minutes}min":
            raise ValueError(
                f"Model tsp config freq ({self.tsp.config.freq}) does not match expected frequency ({self.config.timestep_minutes}min)."
            )

    def _build_timestamps(self) -> pd.DatetimeIndex:
        """Generate deterministic timestamps for building input dataframes."""
        total_length = self.context_length + self.prediction_length
        start_time = pd.Timestamp("2020-01-01 00:00:00")
        return pd.date_range(start=start_time, periods=total_length, freq=self.tsp.freq)

    def _build_past_dataframe(
            self,
            weather_history: np.ndarray,
            control_history: np.ndarray,
            state_history: np.ndarray,
    ) -> pd.DataFrame:
        """Construct the past DataFrame for model input."""
        data = {
            self.tsp.timestamp_column: self._timestamps[:self.context_length],
            "series_id": 0
        }

        for i, var_name in enumerate(self.state_vars):
            data[var_name] = state_history[:, i].astype(np.float32, copy=True)

        for i, var_name in enumerate(self.weather_vars):
            data[var_name] = weather_history[:, i].astype(np.float32, copy=True)

        for i, var_name in enumerate(self.control_vars):
            data[var_name] = control_history[:, i].astype(np.float32, copy=True)

        return pd.DataFrame(data)

    def _build_future_dataframe(
            self,
            weather_forecast: np.ndarray,
            control_plan: np.ndarray,
    ) -> pd.DataFrame:
        """Construct the future DataFrame containing known inputs."""
        data = {
            self.tsp.timestamp_column: self._timestamps[self.context_length:],
            "series_id": 0
        }

        for i, var_name in enumerate(self.weather_vars):
            data[var_name] = weather_forecast[:, i].astype(np.float32, copy=True)

        for i, var_name in enumerate(self.control_vars):
            data[var_name] = control_plan[:, i].astype(np.float32, copy=True)

        return pd.DataFrame(data)

    def _extract_predictions(self, forecast_df: pd.DataFrame) -> Float[np.ndarray, "horizon n_states"]:
        """Extract the forecasted targets as a numpy array."""
        missing = [col for col in self.state_vars if col not in forecast_df.columns]
        if missing:
            raise ValueError(f"Forecast DataFrame is missing required target columns: {missing}")

        preds = forecast_df[self.state_vars].to_numpy(dtype=np.float64)
        return preds

    @jaxtyped(typechecker=beartype)
    def predict(
            self,
            weather_history: Float[np.ndarray, "{self.context_length} {self.config.n_weather}"],
            control_history: Float[np.ndarray, "{self.context_length} {self.config.n_controls}"],
            state_history: Float[np.ndarray, "{self.context_length} {self.config.n_states}"],
            weather_forecast: Float[np.ndarray, "{self.prediction_length} {self.config.n_weather}"],
            control_plan: Float[np.ndarray, "{self.prediction_length} {self.config.n_controls}"],
    ) -> Float[np.ndarray, "{self.prediction_length} {self.config.n_states}"]:
        """
        Run inference using the wrapped TinyTimeMixer model.

        Args:
            weather_history: Shape (history_length, n_weather)
            control_history: Shape (history_length, n_controls)
            state_history: Shape (history_length, n_states)
            weather_forecast: Shape (horizon, n_weather)
            control_plan: Shape (horizon, n_controls)

        Returns:
            np.ndarray: Forecasted next states of shape (horizon, n_states)
        """
        past_df = self._build_past_dataframe(weather_history, control_history, state_history)
        future_df = self._build_future_dataframe(weather_forecast, control_plan)

        forecast_df = self.pipeline(past_df, future_time_series=future_df)
        return self._extract_predictions(forecast_df)
