import logging
import numpy as np
import pandas as pd
from typing import Optional

from hvacbench.config import EnvConfig
from hvacbench.models.base import BaseTTM
from hvacbench.schemas import FloatArray

logger = logging.getLogger(__name__)

class TTMForecasterModel(BaseTTM):
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

        self.timestamp_column = getattr(config, "timestamp_column", "time")
        self.freq = getattr(config, "frequency", "15min")

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

    def _build_timestamps(self) -> pd.DatetimeIndex:
        """Generate deterministic timestamps for building input dataframes."""
        total_length = self.context_length + self.prediction_length
        start_time = pd.Timestamp("2020-01-01 00:00:00")
        return pd.date_range(start=start_time, periods=total_length, freq=self.freq)

    def _build_past_dataframe(
        self,
        weather_history: np.ndarray,
        control_history: np.ndarray,
        state_history: np.ndarray,
    ) -> pd.DataFrame:
        """Construct the past DataFrame for model input."""
        timestamps = self._build_timestamps()[:self.context_length]
        df = pd.DataFrame({self.timestamp_column: timestamps})

        for i, var_name in enumerate(self.state_vars):
            df[var_name] = state_history[:, i]

        for i, var_name in enumerate(self.weather_vars):
            df[var_name] = weather_history[:, i]

        for i, var_name in enumerate(self.control_vars):
            df[var_name] = control_history[:, i]

        df["series_id"] = 0

        # Ensure writable arrays to avoid tsfm dataset read-only errors
        for col in df.columns:
            if col not in (self.timestamp_column, "series_id"):
                df[col] = df[col].astype(np.float32).copy()

        return df

    def _build_future_dataframe(
        self,
        weather_forecast: np.ndarray,
        control_plan: np.ndarray,
    ) -> pd.DataFrame:
        """Construct the future DataFrame containing known inputs."""
        timestamps = self._build_timestamps()[self.context_length:]
        df = pd.DataFrame({self.timestamp_column: timestamps})

        for i, var_name in enumerate(self.weather_vars):
            df[var_name] = weather_forecast[:, i]

        for i, var_name in enumerate(self.control_vars):
            df[var_name] = control_plan[:, i]

        df["series_id"] = 0

        for col in df.columns:
            if col not in (self.timestamp_column, "series_id"):
                df[col] = df[col].astype(np.float32).copy()

        return df

    def _extract_predictions(self, forecast_df: pd.DataFrame) -> FloatArray:
        """Extract the forecasted targets as a numpy array."""
        missing = [col for col in self.state_vars if col not in forecast_df.columns]
        if missing:
            raise ValueError(f"Forecast DataFrame is missing required target columns: {missing}")

        preds: FloatArray = forecast_df[self.state_vars].to_numpy(dtype=np.float64)

        if preds.shape != (self.prediction_length, self.config.n_states):
            raise ValueError(
                f"Expected predictions of shape {(self.prediction_length, self.config.n_states)}, "
                f"but got {preds.shape}"
            )

        return preds

    def predict(
        self,
        weather_history: FloatArray,
        control_history: FloatArray,
        state_history: FloatArray,
        weather_forecast: FloatArray,
        control_plan: FloatArray,
    ) -> FloatArray:
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
