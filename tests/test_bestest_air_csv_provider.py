from pathlib import Path

import numpy as np
import pytest

from hvacbench.config import EnvConfig
from hvacbench.energy_price import EnergyPriceType
from hvacbench.providers.bestest_air import BestestAirCsvProvider


DATA_DIR = Path(__file__).parents[1] / "src" / "hvacbench" / "data"


@pytest.mark.parametrize("energy_price_type", list(EnergyPriceType))
def test_bestest_air_provider_shapes(energy_price_type):
    config = EnvConfig(history_length=4, horizon=8)
    provider = BestestAirCsvProvider(
        config=config,
        energy_price_type=energy_price_type,
    )

    assert provider.get_weather_forecast(0, config.horizon).shape == (
        config.horizon,
        config.n_weather,
    )
    assert provider.get_energy_price_forecast(0, config.horizon).shape == (
        config.horizon,
    )
    assert provider.get_initial_weather_history(config.history_length).shape == (
        config.history_length,
        config.n_weather,
    )
    assert provider.get_initial_control_history(config.history_length).shape == (
        config.history_length,
        config.n_controls,
    )
    assert provider.get_initial_state_history(config.history_length).shape == (
        config.history_length,
        config.n_states,
    )


def test_bestest_air_provider_accepts_explicit_data_paths():
    config = EnvConfig(history_length=4, horizon=8)
    provider = BestestAirCsvProvider(
        config=config,
        building_data_path=DATA_DIR / "bestest_air_default_1y.csv",
        electricity_price_data_path=DATA_DIR / "electricity_prices_bestest_air_1y.csv",
    )

    assert provider.get_weather_forecast(0, config.horizon).shape == (
        config.horizon,
        config.n_weather,
    )
    assert provider.get_energy_price_forecast(0, config.horizon).shape == (
        config.horizon,
    )


def test_bestest_air_provider_uses_configured_variable_order():
    config = EnvConfig(history_length=4, horizon=8)
    provider = BestestAirCsvProvider(config=config)

    first_weather = provider.get_weather_forecast(0, 1)[0]
    assert np.allclose(
        first_weather,
        np.array([0.0, 0.51725, 2.316458333333333, 0.0], dtype=np.float64),
    )

    initial_controls = provider.get_initial_control_history(1)
    assert np.allclose(initial_controls[0], np.array([15.0, 30.0]))


def test_bestest_air_provider_wraps_cyclically():
    config = EnvConfig(history_length=4, horizon=8)
    provider = BestestAirCsvProvider(config=config)

    tail_then_head = provider.get_weather_forecast(-1, 2)
    head = provider.get_weather_forecast(0, 1)

    assert np.allclose(tail_then_head[1], head[0])
    assert np.allclose(provider.get_initial_weather_history(1), tail_then_head[:1])


def test_highly_dynamic_prices_align_to_fifteen_minute_steps():
    config = EnvConfig(history_length=4, horizon=5)
    provider = BestestAirCsvProvider(
        config=config,
        energy_price_type=EnergyPriceType.HIGHLY_DYNAMIC,
    )

    prices = provider.get_energy_price_forecast(0, 5)

    assert np.allclose(prices[:4], prices[0])
    assert not np.isclose(prices[4], prices[0])
