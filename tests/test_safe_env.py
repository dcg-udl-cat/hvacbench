import numpy as np
from hvacbench.safety.control_safety import ControlSafetyFilter
from hvacbench.config import EnvConfig
from hvacbench.providers.mock import MockProvider
from hvacbench.models.mock import MockBuildingModel
from hvacbench.rewards.simple import SimpleReward
from hvacbench.envs.ttm_env import TTMEnv
from hvacbench.envs.safe_env import SafeEnv

def test_safe_env_clipping_and_logic():
    config = EnvConfig(horizon=2)
    provider = MockProvider(config)
    model = MockBuildingModel(config)
    reward = SimpleReward(config)
    base_env = TTMEnv(config=config, provider=provider, reward=reward, model=model)
    safety_filter = ControlSafetyFilter(heating_min=15, heating_max=24, cooling_min=22, cooling_max=30)
    
    env = SafeEnv(base_env, safety_filter)
    env.reset()
    
    # Intentionally bad action
    # 0: heating, 1: cooling
    # action 1: heating > cooling (25 > 20) -> out of bounds too
    # action 2: fine but cooling out of bounds
    action = np.array([
        [24.0, 20.0],
        [20.0, 35.0]
    ])
    
    obs, reward_val, term, trunc, info = env.step(action)
    
    safe_action = info["safe_action"]
    
    # Action 1 expected:
    # bounds apply first: heating -> 24.0, cooling -> 22.0
    # heating (24) > cooling (22) -> cooling becomes max(24, 22) -> 24
    assert safe_action[0, 0] == 24.0
    assert safe_action[0, 1] == 24.0
    
    # Action 2 expected:
    # bounds: heating -> 20.0, cooling -> 30.0
    # heating <= cooling holds.
    assert safe_action[1, 0] == 20.0
    assert safe_action[1, 1] == 30.0
