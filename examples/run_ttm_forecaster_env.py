import logging
from pathlib import Path

from hvacbench.config import EnvConfig
from hvacbench.envs.ttm_env import TTMEnv
from hvacbench.envs.safe_env import SafeEnv
from hvacbench.models.ttm import TTM
from hvacbench.providers.mock import MockProvider
from hvacbench.rewards.simple import SimpleReward
from hvacbench.safety.control_safety import ControlSafetyFilter

logging.basicConfig(level=logging.INFO)


def main():
    config = EnvConfig()

    provider = MockProvider(config=config)
    reward = SimpleReward(config=config)
    safety_filter = ControlSafetyFilter()

    model_path = Path("gft/ttm4hvac")

    model = TTM(
        config=config,
        model_path=str(model_path),
    )

    base_env = TTMEnv(
        config=config,
        provider=provider,
        reward=reward,
        model=model,
    )

    env = SafeEnv(
        env=base_env,
        safety_filter=safety_filter,
    )

    obs, info = env.reset()

    action = provider.get_random_action()

    next_obs, reward_val, terminated, truncated, step_info = env.step(action)
    logging.info(f"Stepped. Reward: {reward_val}, Terminated: {terminated}")

if __name__ == "__main__":
    main()

