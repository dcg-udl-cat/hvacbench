from collections.abc import Sequence
from importlib.resources import files
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Float, jaxtyped

from hvacbench.config import STEP_PERIOD_SECONDS, EnvConfig, TTMVariables
from hvacbench.energy_price import EnergyPriceType
from hvacbench.providers.base import BaseProvider
from hvacbench.schemas import FloatArray


BUILDING_DATA_FILENAME = "bestest_air_default_1y.csv"
ELECTRICITY_PRICE_DATA_FILENAME = "electricity_prices_bestest_air_1y.csv"
BESTEST_AIR_YEAR_TIMESTEPS = 365 * 24 * 3600 // STEP_PERIOD_SECONDS


class BestestAirCsvProvider(BaseProvider):
    """CSV-backed provider for cyclic bestest_air data."""

    _DATA_PACKAGE = "hvacbench.data"

    def __init__(
        self,
        config: EnvConfig,
        energy_price_type: EnergyPriceType = EnergyPriceType.DYNAMIC,
        variables: TTMVariables = TTMVariables(),
        building_data_path: str | PathLike[str] | None = None,
        electricity_price_data_path: str | PathLike[str] | None = None,
    ) -> None:
        self.config = config
        self.energy_price_type = EnergyPriceType(energy_price_type)
        self.variables = variables

        building_columns = (
            *self.variables.weather_names,
            *self.variables.control_names,
            *self.variables.state_names,
        )
        building_df = self._read_csv(
            path=building_data_path,
            default_filename=BUILDING_DATA_FILENAME,
            usecols=building_columns,
        )
        price_df = self._read_csv(
            path=electricity_price_data_path,
            default_filename=ELECTRICITY_PRICE_DATA_FILENAME,
            usecols=[self.energy_price_type.value],
        )

        self._weather: FloatArray = self._extract_matrix(
            building_df,
            self.variables.weather_names,
        )
        self._controls: FloatArray = self._extract_matrix(
            building_df,
            self.variables.control_names,
        )
        self._states: FloatArray = self._extract_matrix(
            building_df,
            self.variables.state_names,
        )
        self._energy_prices: FloatArray = np.asarray(
            price_df[self.energy_price_type.value].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )

    @jaxtyped(typechecker=beartype)
    def get_weather_forecast(
        self,
        t: int,
        horizon: int,
    ) -> Float[np.ndarray, "{horizon} {self.config.n_weather}"]:
        return self._cyclic_rows(self._weather, t, horizon)

    @jaxtyped(typechecker=beartype)
    def get_energy_price_forecast(
        self,
        t: int,
        horizon: int,
    ) -> Float[np.ndarray, "{horizon}"]:
        indices = self._cyclic_indices(t, horizon)
        return np.asarray(self._energy_prices[indices], dtype=np.float64).copy()

    @jaxtyped(typechecker=beartype)
    def get_initial_weather_history(
        self,
        history_length: int,
        start_timestep: int = 0,
    ) -> Float[np.ndarray, "{history_length} {self.config.n_weather}"]:
        return self.get_weather_forecast(
            start_timestep - history_length,
            history_length,
        )

    @jaxtyped(typechecker=beartype)
    def get_initial_control_history(
        self,
        history_length: int,
        start_timestep: int = 0,
    ) -> Float[np.ndarray, "{history_length} {self.config.n_controls}"]:
        return self._cyclic_rows(
            self._controls,
            start_timestep - history_length,
            history_length,
        )

    @jaxtyped(typechecker=beartype)
    def get_initial_state_history(
        self,
        history_length: int,
        start_timestep: int = 0,
    ) -> Float[np.ndarray, "{history_length} {self.config.n_states}"]:
        return self._cyclic_rows(
            self._states,
            start_timestep - history_length,
            history_length,
        )

    @jaxtyped(typechecker=beartype)
    def get_random_action(
        self,
    ) -> Float[np.ndarray, "{self.config.horizon} {self.config.n_controls}"]:
        action = np.zeros(
            (self.config.horizon, self.config.n_controls),
            dtype=np.float64,
        )
        action[:, 0] = np.random.uniform(18.0, 22.0, size=self.config.horizon)
        action[:, 1] = np.random.uniform(22.0, 26.0, size=self.config.horizon)
        return action

    def _cyclic_rows(
        self,
        values: FloatArray,
        t: int,
        horizon: int,
    ) -> FloatArray:
        indices = self._cyclic_indices(t, horizon)
        return np.asarray(values[indices], dtype=np.float64).copy()

    def _cyclic_indices(self, t: int, length: int) -> np.ndarray:
        return np.mod(
            np.arange(t, t + length, dtype=np.int64), BESTEST_AIR_YEAR_TIMESTEPS
        )

    @staticmethod
    def _extract_matrix(df: pd.DataFrame, columns: Sequence[str]) -> FloatArray:
        return np.asarray(
            df.loc[:, list(columns)].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )

    @classmethod
    def _read_csv(
        cls,
        path: str | PathLike[str] | None,
        default_filename: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        read_kwargs = {"nrows": BESTEST_AIR_YEAR_TIMESTEPS, **kwargs}
        if path is not None:
            return pd.read_csv(path, **read_kwargs)

        resource = files(cls._DATA_PACKAGE).joinpath(default_filename)
        with resource.open("rb") as csv_file:
            return pd.read_csv(csv_file, **read_kwargs)
