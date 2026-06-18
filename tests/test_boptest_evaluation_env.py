import numpy as np
import pytest
from jaxtyping import TypeCheckError

from hvacbench.boptest.mock import MockBoptestClient
from hvacbench.config import EnvConfig
from hvacbench.envs import BoptestEvaluationEnv
from hvacbench.rewards.simple import SimpleReward


def make_env(
    *,
    config: EnvConfig | None = None,
    main_client: MockBoptestClient | None = None,
    start_day: int = 0,
) -> BoptestEvaluationEnv:
    config = config or EnvConfig()
    return BoptestEvaluationEnv(
        reward=SimpleReward(config),
        config=config,
        main_client=main_client or MockBoptestClient(testid="main"),
        start_day=start_day,
    )


def test_construct_and_get_obs_shapes() -> None:
    env = make_env()
    obs = env.get_obs()

    assert obs.weather_history.shape == (env.config.history_length, 4)
    assert obs.control_history.shape == (env.config.history_length, 2)
    assert obs.state_history.shape == (env.config.history_length, 2)
    assert obs.weather_forecast.shape == (env.config.horizon, 4)
    assert obs.energy_price_forecast.shape == (env.config.horizon,)


def test_step_commits_first_control_without_shadow_rollout() -> None:
    main_client = MockBoptestClient(testid="main")
    env = make_env(main_client=main_client)
    control_plan = np.tile(
        np.array([[21.0, 24.0]], dtype=np.float64),
        (env.config.horizon, 1),
    )
    weather_before = env.get_obs().weather_forecast[0].copy()

    _obs, reward, terminated, truncated, info = env.step(control_plan)

    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert len(main_client.advance_calls) == env.config.history_length + 1
    assert set(info) == {
        "predicted_states",
        "realized_state",
        "applied_control",
        "control_plan",
        "reward_horizon_steps",
        "reward_components",
    }
    assert info["predicted_states"].shape == (1, env.config.n_states)
    assert info["realized_state"].shape == (env.config.n_states,)
    assert info["reward_horizon_steps"] == 1
    assert np.allclose(info["applied_control"], control_plan[0])

    assert np.allclose(env.weather_history[-1], weather_before)
    assert np.allclose(env.control_history[-1], control_plan[0])
    assert env.state_history[-1, 0] == pytest.approx(
        20.0 + env.config.history_length + 1
    )
    assert env.state_history[-1, 1] == pytest.approx(130.0)


def test_truncates_from_configured_episode_length() -> None:
    config = EnvConfig(history_length=2, horizon=2, total_simulation_seconds=900)
    env = make_env(config=config)
    control_plan = np.tile(
        np.array([[21.0, 24.0]], dtype=np.float64),
        (config.horizon, 1),
    )

    _obs, _reward, terminated, truncated, _info = env.step(control_plan)

    assert terminated is False
    assert truncated is True


def test_invalid_action_shape() -> None:
    config = EnvConfig(history_length=2, horizon=2)
    env = make_env(config=config)

    with pytest.raises(TypeCheckError):
        env.step(np.ones((1, config.n_controls), dtype=np.float64))


def test_start_day_sets_initialize_start_time() -> None:
    main_client = MockBoptestClient(testid="main")
    env = make_env(main_client=main_client, start_day=4)
    expected_start_time_seconds = 4 * 24 * 3600

    assert env.start_time_seconds == expected_start_time_seconds
    assert main_client.initialize_calls[-1] == {
        "start_time_seconds": expected_start_time_seconds,
        "warmup_period_seconds": 0,
    }
