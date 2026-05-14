import pytest
import numpy as np
from hvacbench.config import EnvConfig
from hvacbench.providers.mock import MockProvider
from hvacbench.models.mock import MockBuildingModel
from hvacbench.rewards.simple import SimpleReward
from hvacbench.envs.ttm_env import TTMEnv

@pytest.fixture
def config():
    return EnvConfig(history_length=1536, horizon=96)

@pytest.fixture
def ttm_env(config):
    provider = MockProvider(config)
    model = MockBuildingModel(config)
    reward = SimpleReward(config)
    return TTMEnv(config=config, provider=provider, reward=reward, model=model)

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

def test_invalid_action_shape(ttm_env, config):
    ttm_env.reset()
    action = np.ones((10, config.n_controls)) # Invalid horizon length
    with pytest.raises(ValueError, match="Invalid shape for action"):
        ttm_env.step(action)
