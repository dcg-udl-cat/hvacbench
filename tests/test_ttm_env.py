import pytest
import numpy as np
from jaxtyping import TypeCheckError

from hvacbench.config import EnvConfig, TTMVariables
from hvacbench.providers.mock import MockProvider
from hvacbench.models.mock import MockTTM
from hvacbench.rewards.simple import SimpleReward
from hvacbench.envs.ttm_env import TTMEnv


class RecordingMockTTM(MockTTM):
    def __init__(
        self,
        config: EnvConfig,
        context_length: int,
        prediction_length: int | None = None,
    ):
        super().__init__(
            config,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        self.last_history_shapes = None
        self.last_weather_forecast = None
        self.last_control_plan = None

    def predict(
        self,
        weather_history,
        control_history,
        state_history,
        weather_forecast,
        control_plan,
    ):
        self.last_history_shapes = (
            weather_history.shape,
            control_history.shape,
            state_history.shape,
        )
        self.last_weather_forecast = weather_forecast.copy()
        self.last_control_plan = control_plan.copy()
        return super().predict(
            weather_history,
            control_history,
            state_history,
            weather_forecast,
            control_plan,
        )


@pytest.fixture
def config():
    return EnvConfig(history_length=1536, horizon=96)


@pytest.fixture
def ttm_env(config):
    provider = MockProvider(config)
    model = MockTTM(config)
    reward = SimpleReward(config)
    return TTMEnv(
        config=config,
        provider=provider,
        reward=reward,
        model=model,
        variables=TTMVariables(),
    )


def test_initialization_and_reset(ttm_env, config):
    obs, info = ttm_env.reset()
    assert obs.weather_history.shape == (config.history_length, config.n_weather)
    assert obs.control_history.shape == (config.history_length, config.n_controls)
    assert obs.state_history.shape == (config.history_length, config.n_states)
    assert obs.weather_forecast.shape == (config.horizon, config.n_weather)
    assert obs.energy_price_forecast.shape == (config.horizon,)
    assert info == {}


def test_step(ttm_env, config):
    ttm_env.reset()
    action = np.ones((config.horizon, config.n_controls)) * 22.0
    next_obs, reward, terminated, truncated, info = ttm_env.step(action)

    assert isinstance(reward, float)
    assert not terminated

    # Check histories length
    assert ttm_env.weather_history.shape == (config.history_length, config.n_weather)
    assert ttm_env.control_history.shape == (config.history_length, config.n_controls)
    assert ttm_env.state_history.shape == (config.history_length, config.n_states)

    # Latest control should be appended to history
    assert np.allclose(ttm_env.control_history[-1], action[0])


def test_ttm_env_truncates_from_config_not_provider_data_length():
    config = EnvConfig(history_length=3, horizon=2, total_simulation_seconds=900)
    provider = MockProvider(config)
    model = MockTTM(config)
    reward = SimpleReward(config)
    env = TTMEnv(
        config=config,
        provider=provider,
        reward=reward,
        model=model,
        variables=TTMVariables(),
    )

    action = np.ones((config.horizon, config.n_controls)) * 22.0
    _next_obs, _reward, terminated, truncated, _info = env.step(action)

    assert not hasattr(provider, "total_timesteps")
    assert not terminated
    assert truncated


def test_get_random_control_plan_shape(ttm_env, config):
    control_plan = ttm_env.get_random_control_plan()

    assert control_plan.shape == (config.horizon, config.n_controls)


def test_invalid_action_shape(ttm_env, config):
    ttm_env.reset()
    action = np.ones((10, config.n_controls))  # Invalid horizon length
    with pytest.raises(TypeCheckError):
        ttm_env.step(action)


@pytest.mark.parametrize(
    ("observation_history_length", "model_context_length", "buffer_length"),
    [
        (3, 5, 5),
        (6, 4, 6),
    ],
)
def test_observation_history_and_model_context_are_decoupled(
    observation_history_length,
    model_context_length,
    buffer_length,
):
    config = EnvConfig(history_length=observation_history_length, horizon=2)
    provider = MockProvider(config)
    model = RecordingMockTTM(config, context_length=model_context_length)
    reward = SimpleReward(config)
    env = TTMEnv(
        config=config,
        provider=provider,
        reward=reward,
        model=model,
        variables=TTMVariables(),
    )

    obs, _ = env.reset()
    assert obs.weather_history.shape == (observation_history_length, config.n_weather)
    assert obs.control_history.shape == (observation_history_length, config.n_controls)
    assert obs.state_history.shape == (observation_history_length, config.n_states)

    assert env.weather_history.shape == (buffer_length, config.n_weather)
    assert env.control_history.shape == (buffer_length, config.n_controls)
    assert env.state_history.shape == (buffer_length, config.n_states)

    action = np.ones((config.horizon, config.n_controls)) * 22.0
    next_obs, *_ = env.step(action)

    assert model.last_history_shapes == (
        (model_context_length, config.n_weather),
        (model_context_length, config.n_controls),
        (model_context_length, config.n_states),
    )
    assert next_obs.weather_history.shape == (
        observation_history_length,
        config.n_weather,
    )
    assert next_obs.control_history.shape == (
        observation_history_length,
        config.n_controls,
    )
    assert next_obs.state_history.shape == (observation_history_length, config.n_states)


def test_env_horizon_must_not_exceed_model_prediction_length():
    config = EnvConfig(history_length=3, horizon=4)
    provider = MockProvider(config)
    model = MockTTM(config, prediction_length=3)
    reward = SimpleReward(config)

    with pytest.raises(ValueError, match="horizon=.*prediction_length"):
        TTMEnv(
            config=config,
            provider=provider,
            reward=reward,
            model=model,
            variables=TTMVariables(),
        )


def test_short_env_horizon_is_padded_for_model_prediction_length():
    config = EnvConfig(history_length=3, horizon=2)
    provider = MockProvider(config)
    model = RecordingMockTTM(config, context_length=3, prediction_length=4)
    reward = SimpleReward(config)
    env = TTMEnv(
        config=config,
        provider=provider,
        reward=reward,
        model=model,
        variables=TTMVariables(),
    )

    control_plan = np.array(
        [
            [20.0, 24.0],
            [21.0, 25.0],
        ],
        dtype=np.float64,
    )

    _next_obs, _reward, _terminated, _truncated, info = env.step(control_plan)

    assert model.last_weather_forecast.shape == (4, config.n_weather)
    assert model.last_control_plan.shape == (4, config.n_controls)
    assert np.allclose(model.last_control_plan[: config.horizon], control_plan)
    assert np.allclose(model.last_control_plan[config.horizon :], control_plan[-1])
    assert info["predicted_states"].shape == (config.horizon, config.n_states)
    assert info["control_plan"].shape == (config.horizon, config.n_controls)
