import numpy as np
import pytest

from hvacbench.boptest.bestest_air import BestestAir
from hvacbench.boptest.mock import MockBoptestClient


def test_control_conversion_celsius_to_kelvin() -> None:
    testcase = BestestAir()
    client = MockBoptestClient()
    inputs = client.control_row_to_inputs(
        testcase,
        np.array([21.0, 24.0], dtype=np.float64)
    )

    assert inputs["con_oveTSetHea_u"] == pytest.approx(294.15)
    assert inputs["con_oveTSetHea_activate"] == 1.0
    assert inputs["con_oveTSetCoo_u"] == pytest.approx(297.15)
    assert inputs["con_oveTSetCoo_activate"] == 1.0


def test_state_extraction_kelvin_to_celsius_and_hvac_power_sum() -> None:
    testcase = BestestAir()
    client = MockBoptestClient()
    state = client.extract_state_from_values(
        testcase,
        {
            "zon_reaTRooAir_y": 295.15,
            "fcu_reaPCoo_y": 100.0,
            "fcu_reaPFan_y": 20.0,
            "fcu_reaPHea_y": 300.0,
        }
    )

    assert state.shape == (2,)
    assert state[0] == pytest.approx(22.0)
    assert state[1] == pytest.approx(420.0)


def test_weather_conversions() -> None:
    testcase = BestestAir()
    client = MockBoptestClient()
    outdoor_c = client.convert_weather_point(
        testcase,
        "TDryBul",
        np.array([283.15, 284.15], dtype=np.float64),
    )
    humidity_percent = client.convert_weather_point(
        testcase,
        "relHum",
        np.array([0.4, 0.5], dtype=np.float64),
    )

    assert np.allclose(outdoor_c, np.array([10.0, 11.0]))
    assert np.allclose(humidity_percent, np.array([40.0, 50.0]))


def test_electricity_price_point_selection() -> None:
    assert (
        BestestAir(energy_price_type="constant").energy_price_forecast_point()
        == "PriceElectricPowerConstant"
    )
    assert (
        BestestAir(energy_price_type="dynamic").energy_price_forecast_point()
        == "PriceElectricPowerDynamic"
    )
    assert (
        BestestAir(energy_price_type="highly_dynamic").energy_price_forecast_point()
        == "PriceElectricPowerHighlyDynamic"
    )


def test_missing_state_point_raises_clear_error() -> None:
    testcase = BestestAir()
    client = MockBoptestClient()
    with pytest.raises(KeyError, match="missing required bestest_air"):
        client.extract_state_from_values(
            testcase,
            {
                "zon_reaTRooAir_y": 295.15,
                "fcu_reaPCoo_y": 100.0,
                "fcu_reaPFan_y": 20.0,
            }
        )
