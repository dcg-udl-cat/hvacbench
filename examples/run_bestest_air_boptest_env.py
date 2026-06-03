import numpy as np

from hvacbench.boptest.bestest_air import BestestAir
from hvacbench.config import EnvConfig
from hvacbench.envs.boptest_env import BoptestEnv
from hvacbench.envs.safe_env import SafeEnv
from hvacbench.rewards.simple import SimpleReward
from hvacbench.safety.control_safety import ControlSafetyFilter


def main() -> None:
    config = EnvConfig(
        history_length=8,
        horizon=96,
        total_simulation_seconds=3 * 15 * 60,
    )
    testcase = BestestAir(
        base_url="http://127.0.0.1",
        energy_price_type="dynamic",
    )
    reward = SimpleReward(config=config)

    with BoptestEnv(
        reward=reward,
        config=config,
        testcase=testcase,
    ) as base_env:
        use_safe_env = False
        env = (
            SafeEnv(base_env, safety_filter=ControlSafetyFilter())
            if use_safe_env
            else base_env
        )

        obs = env.get_obs()
        print(
            "Initial BOPTEST time: "
            f"{base_env.current_time_seconds} simulated seconds"
        )

        control_plan = np.tile(
            np.array([[21.0, 24.0]], dtype=np.float64),
            (config.horizon, 1),
        )

        for step_number in range(1, 4):
            obs, reward_value, terminated, truncated, info = env.step(control_plan)
            latest_state = obs.state_history[-1]

            print(f"Step: {step_number}")
            print(f"  Reward: {reward_value:.3f}")
            print(f"  Latest room temperature: {latest_state[0]:.2f} C")
            print(f"  Latest HVAC power: {latest_state[1]:.2f} W")
            print(f"  Applied control: {info['applied_control']}")
            print(f"  KPIs: {base_env.main_client.get_kpis()}")

            if terminated or truncated:
                break

    print("Stopped BOPTEST testcases.")


if __name__ == "__main__":
    main()
