from hvacbench.config import EnvConfig
from hvacbench.energy_price import EnergyPriceType
from hvacbench.envs import BoptestRolloutEnv
from hvacbench.rewards.simple import SimpleReward
from tqdm import tqdm


def main() -> None:
    config = EnvConfig(
        history_length=8, horizon=8, total_simulation_seconds=7 * 24 * 3600
    )

    reward = SimpleReward(config=config)

    env = BoptestRolloutEnv(
        reward=reward,
        config=config,
        energy_price_type=EnergyPriceType.DYNAMIC,
    )

    obs = env.get_obs()

    control_plan = env.get_random_control_plan()

    total_steps = config.total_simulation_seconds // config.step_period_seconds

    for i in tqdm(range(int(total_steps))):
        obs, reward_value, terminated, truncated, info = env.step(control_plan)
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
