from hvacbench.config import EnvConfig, TTMVariables
from hvacbench.providers.mock import MockProvider
from hvacbench.models.mock import MockTTM
from hvacbench.rewards.simple import SimpleReward
from hvacbench.envs.ttm_env import TTMEnv
from hvacbench.envs.safe_env import SafeEnv
from hvacbench.safety.control_safety import ControlSafetyFilter
import numpy as np

def main():
    config = EnvConfig(
        history_length=8,
        horizon=96,
        total_simulation_seconds= 7 * 24 * 3600
    )
    provider = MockProvider(config=config, seed=42)
    model = MockTTM(config=config)
    reward = SimpleReward(config=config)
    variables = TTMVariables()
    
    base_env = TTMEnv(
        config=config,
        provider=provider,
        reward=reward,
        model=model,
        variables=variables,
    )
    safety_filter = ControlSafetyFilter()
    env = SafeEnv(base_env, safety_filter=safety_filter)
    
    obs, info = env.reset()
    
    print("Starting simulation...\n")
    
    for step_num in range(1, 11):
        # A simple constant control plan for 96 steps
        action = np.zeros((config.horizon, config.n_controls))
        action[:, 0] = 22.0  # Heating setpoint
        action[:, 1] = 24.0  # Cooling setpoint
        
        obs, reward_val, term, trunc, info_dict = env.step(action)
        
        applied_control = info_dict["applied_control"]
        predicted_states = info_dict["predicted_states"]
        latest_temp = predicted_states[0, 0]
        latest_power = predicted_states[0, 1]
        
        print(f"Step: {step_num}")
        print(f"  Applied Control: {applied_control}")
        print(f"  Latest Room Temp: {latest_temp:.2f} C")
        print(f"  Latest HVAC Power: {latest_power:.2f} W")
        print(f"  Reward: {reward_val:.2f}\n")

if __name__ == "__main__":
    main()
