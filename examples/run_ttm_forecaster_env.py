import logging
from pathlib import Path

from hvacbench.config import EnvConfig
from hvacbench.energy_price import EnergyPriceType
from hvacbench.envs.ttm_env import TTMEnv
from hvacbench.envs.safe_env import SafeEnv
from hvacbench.models.ttm import TTM
from hvacbench.providers.bestest_air import BestestAirCsvProvider
from hvacbench.rewards.simple import SimpleReward
from hvacbench.safety.control_safety import ControlSafetyFilter

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def main():
    config = EnvConfig(
        history_length=8, horizon=8, total_simulation_seconds=7 * 24 * 3600
    )

    provider = BestestAirCsvProvider(
        config=config,
        energy_price_type=EnergyPriceType.DYNAMIC,
    )
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

    control_plan = env.get_random_control_plan()

    total_steps = config.total_simulation_seconds // config.step_period_seconds
    for i in tqdm(range(int(total_steps))):
        next_obs, reward_val, terminated, truncated, step_info = env.step(control_plan)

    # logging.info(f"Stepped. Reward: {reward_val}, Terminated: {terminated}")


if __name__ == "__main__":
    main()
