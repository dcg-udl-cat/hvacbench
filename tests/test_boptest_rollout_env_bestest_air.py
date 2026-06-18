from typing import Any

import numpy as np
import pytest

from hvacbench.boptest.client import BoptestClient
from hvacbench.boptest.mock import MockBoptestClient
from hvacbench.config import EnvConfig
from hvacbench.envs import BoptestRolloutEnv
from hvacbench.rewards.simple import SimpleReward


def make_env(
    *,
    main_client: MockBoptestClient | None = None,
    rollout_client: MockBoptestClient | None = None,
    start_day: int = 0,
) -> BoptestRolloutEnv:
    config = EnvConfig()
    reward = SimpleReward(config)
    return BoptestRolloutEnv(
        reward=reward,
        config=config,
        main_client=main_client or MockBoptestClient(testid="main"),
        rollout_client=rollout_client or MockBoptestClient(testid="rollout"),
        start_day=start_day,
    )


def test_refuses_clients_with_same_testid() -> None:
    config = EnvConfig()
    with pytest.raises(ValueError, match="different testids"):
        BoptestRolloutEnv(
            reward=SimpleReward(config),
            config=config,
            main_client=MockBoptestClient(testid="same"),
            rollout_client=MockBoptestClient(testid="same"),
        )


def test_construct_and_get_obs_shapes() -> None:
    env = make_env()
    obs = env.get_obs()

    assert obs.weather_history.shape == (env.config.history_length, 4)
    assert obs.control_history.shape == (env.config.history_length, 2)
    assert obs.state_history.shape == (env.config.history_length, 2)
    assert obs.weather_forecast.shape == (env.config.horizon, 4)
    assert obs.energy_price_forecast.shape == (env.config.horizon,)


def test_start_day_sets_boptest_initialize_start_time() -> None:
    main_client = MockBoptestClient(testid="main")
    rollout_client = MockBoptestClient(testid="rollout")
    env = make_env(
        main_client=main_client,
        rollout_client=rollout_client,
        start_day=3,
    )
    expected_start_time_seconds = 3 * 24 * 3600

    assert env.start_time_seconds == expected_start_time_seconds
    assert main_client.initialize_calls[-1] == {
        "start_time_seconds": expected_start_time_seconds,
        "warmup_period_seconds": 0,
    }
    assert rollout_client.initialize_calls[-1] == {
        "start_time_seconds": expected_start_time_seconds,
        "warmup_period_seconds": 0,
    }
    assert env.current_time_seconds == (
        expected_start_time_seconds
        + env.config.history_length * env.testcase.step_period_seconds
    )
    assert main_client.results_calls[-1]["start_time_seconds"] == (
        expected_start_time_seconds
    )
    assert main_client.results_calls[-1]["final_time_seconds"] == env.current_time_seconds


def test_boptest_start_day_must_be_non_negative() -> None:
    config = EnvConfig()

    with pytest.raises(ValueError, match="start_day"):
        BoptestRolloutEnv(
            reward=SimpleReward(config),
            config=config,
            main_client=MockBoptestClient(testid="main"),
            rollout_client=MockBoptestClient(testid="rollout"),
            start_day=-1,
        )


def test_get_random_control_plan_shape() -> None:
    env = make_env()
    control_plan = env.get_random_control_plan()

    assert control_plan.shape == (env.config.horizon, env.config.n_controls)


def test_forecast_conversions_and_request_length() -> None:
    main_client = MockBoptestClient(testid="main")
    env = make_env(main_client=main_client)
    obs = env.get_obs()

    assert obs.weather_forecast[0, 0] == pytest.approx(10.0)
    assert obs.weather_forecast[0, 1] == pytest.approx(50.0)
    assert obs.energy_price_forecast[0] == pytest.approx(0.10)
    assert main_client.forecast_calls[-2]["horizon_seconds"] == 95 * 900
    assert main_client.forecast_calls[-1]["horizon_seconds"] == 95 * 900


def test_step_rollout_sync_commit_first_and_history_updates() -> None:
    main_client = MockBoptestClient(testid="main")
    rollout_client = MockBoptestClient(testid="rollout")
    env = make_env(main_client=main_client, rollout_client=rollout_client)

    control_plan = np.tile(
        np.array([[21.0, 24.0]], dtype=np.float64),
        (env.config.horizon, 1),
    )
    weather_before = env.get_obs().weather_forecast[0].copy()

    _next_obs, reward, terminated, truncated, info = env.step(control_plan)

    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert len(rollout_client.initialize_calls) == 1
    assert len(rollout_client.advance_calls) == env.config.history_length + 96
    assert len(main_client.advance_calls) == env.config.history_length + 1
    assert set(info) == {
        "predicted_states",
        "applied_control",
        "control_plan",
        "reward_components",
    }
    assert np.allclose(info["applied_control"], control_plan[0])

    assert env.weather_history.shape == (env.config.history_length, 4)
    assert env.control_history.shape == (env.config.history_length, 2)
    assert env.state_history.shape == (env.config.history_length, 2)
    assert np.allclose(env.weather_history[-1], weather_before)
    assert np.allclose(env.control_history[-1], control_plan[0])
    assert env.state_history[-1, 0] == pytest.approx(
        20.0 + env.config.history_length + 1
    )
    assert env.state_history[-1, 1] == pytest.approx(130.0)


def test_second_step_syncs_one_committed_control_before_candidate_plan() -> None:
    main_client = MockBoptestClient(testid="main")
    rollout_client = MockBoptestClient(testid="rollout")
    env = make_env(main_client=main_client, rollout_client=rollout_client)

    first_plan = np.tile(
        np.array([[21.0, 24.0]], dtype=np.float64),
        (env.config.horizon, 1),
    )
    second_plan = np.tile(
        np.array([[22.0, 25.0]], dtype=np.float64),
        (env.config.horizon, 1),
    )

    env.step(first_plan)
    _next_obs, _reward, _terminated, _truncated, info = env.step(second_plan)

    assert len(rollout_client.initialize_calls) == 2
    assert len(rollout_client.advance_calls) == env.config.history_length + 1 + 96
    assert set(info) == {
        "predicted_states",
        "applied_control",
        "control_plan",
        "reward_components",
    }

    replayed_control_index = env.config.history_length
    first_candidate_index = replayed_control_index + 1
    assert rollout_client.advance_calls[replayed_control_index][
        "con_oveTSetHea_u"
    ] == pytest.approx(
        294.15
    )
    assert rollout_client.advance_calls[replayed_control_index][
        "con_oveTSetCoo_u"
    ] == pytest.approx(
        297.15
    )
    assert rollout_client.advance_calls[first_candidate_index][
        "con_oveTSetHea_u"
    ] == pytest.approx(
        295.15
    )
    assert rollout_client.advance_calls[first_candidate_index][
        "con_oveTSetCoo_u"
    ] == pytest.approx(
        298.15
    )
    assert len(main_client.advance_calls) == env.config.history_length + 2


class FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self._body = body
        self.text = repr(body)

    def json(self) -> dict[str, Any]:
        return self._body


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def request(
        self,
        method: str,
        url: str,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        del method, url, json, timeout
        self.calls += 1
        if self.calls == 1:
            return FakeResponse(200, {"testid": "selected"})
        return FakeResponse(
            200,
            {
                "status": 400,
                "message": "BOPTEST failed",
                "payload": {},
            },
        )

    def close(self) -> None:
        pass


def test_boptest_client_raises_api_error_for_non_200_boptest_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("requests.Session", FakeSession)
    client = BoptestClient(base_url="http://example.test")

    with pytest.raises(RuntimeError, match="BOPTEST API error"):
        client.get_inputs()
