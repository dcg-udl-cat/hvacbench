from typing import Any, Optional

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from hvacbench.boptest.base import BaseBoptestClient
from hvacbench.boptest.testcase import BoptestTestcase
from hvacbench.config import EnvConfig
from hvacbench.energy_price import EnergyPriceType
from hvacbench.envs.boptest_base_env import BaseBoptestEnv
from hvacbench.rewards.base import RewardStrategy
from hvacbench.schemas import Observation, StepReturn


class BoptestEvaluationEnv(BaseBoptestEnv):
    """Single-client BOPTEST environment for realized policy evaluation.

    The action contract matches TTMEnv and BoptestRolloutEnv: policies still
    emit a full horizon plan. Only the first control row is applied to BOPTEST,
    and the reward is computed on the realized one-step transition.
    """

    def __init__(
        self,
        reward: RewardStrategy,
        config: EnvConfig,
        energy_price_type: EnergyPriceType = EnergyPriceType.DYNAMIC,
        testcase: Optional[BoptestTestcase] = None,
        main_client: Optional[BaseBoptestClient] = None,
        start_day: int = 0,
    ) -> None:
        super().__init__(
            reward=reward,
            config=config,
            energy_price_type=energy_price_type,
            testcase=testcase,
            main_client=main_client,
            start_day=start_day,
        )

        try:
            self.reset()
        except Exception:
            self.close()
            raise

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self._reset_main_client_context()
        return self.get_obs(), {
            "boptest_current_time_seconds": self.current_time_seconds,
        }

    @jaxtyped(typechecker=beartype)
    def step(
        self,
        control_plan: Float[
            np.ndarray,
            "{self.config.horizon} {self.config.n_controls}",
        ],
    ) -> StepReturn:
        current_obs = self.get_obs()
        first_control = control_plan[0].copy()
        main_values = self.main_client.advance(
            self._control_row_to_boptest_inputs(first_control)
        ).values
        realized_state = self._extract_state_from_payload(main_values)
        realized_states = realized_state.reshape(1, -1)
        realized_control_plan = first_control.reshape(1, -1)

        info: dict[str, Any] = {
            "predicted_states": realized_states.copy(),
            "realized_state": realized_state.copy(),
            "applied_control": first_control.copy(),
            "control_plan": control_plan,
            "reward_horizon_steps": 1,
        }

        reward = self.reward.compute_reward(
            predicted_states=realized_states,
            control_plan=realized_control_plan,
            weather_forecast=current_obs.weather_forecast[:1],
            energy_price_forecast=current_obs.energy_price_forecast[:1],
            current_obs=current_obs,
            info=info,
        )

        self._append_to_histories(
            next_weather=current_obs.weather_forecast[0].copy(),
            next_control=first_control,
            next_state=realized_state,
        )

        self.current_time_seconds += self.testcase.step_period_seconds
        self.elapsed_episode_seconds += self.testcase.step_period_seconds

        truncated = self.elapsed_episode_seconds >= self.config.total_simulation_seconds
        return self.get_obs(), float(reward), False, truncated, info

    def __enter__(self) -> "BoptestEvaluationEnv":
        return self


__all__ = ["BoptestEvaluationEnv"]
