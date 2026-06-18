from typing import Any, Optional

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from hvacbench.boptest.base import BaseBoptestClient
from hvacbench.boptest.client import BoptestClient
from hvacbench.boptest.testcase import BoptestTestcase
from hvacbench.config import EnvConfig
from hvacbench.energy_price import EnergyPriceType
from hvacbench.envs.boptest_base_env import BaseBoptestEnv
from hvacbench.rewards.base import RewardStrategy
from hvacbench.schemas import FloatArray, Observation, StepReturn


class BoptestRolloutEnv(BaseBoptestEnv):
    """BOPTEST backend for receding-horizon rollout evaluation.

    This environment uses one committed BOPTEST simulator plus one shadow
    simulator for evaluating the proposed horizon control trajectory.
    """

    def __init__(
        self,
        reward: RewardStrategy,
        config: EnvConfig,
        energy_price_type: EnergyPriceType = EnergyPriceType.DYNAMIC,
        testcase: Optional[BoptestTestcase] = None,
        main_client: Optional[BaseBoptestClient] = None,
        rollout_client: Optional[BaseBoptestClient] = None,
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
        self.rollout_client = rollout_client or BoptestClient(self.testcase.base_url)
        self._validate_clients()

        self._committed_controls: list[FloatArray] = []

        try:
            self.reset()
        except Exception:
            self.close()
            raise

    def _validate_clients(self) -> None:
        if self.main_client.testid == self.rollout_client.testid:
            raise ValueError(
                "Main BOPTEST client and rollout client must have different testids."
            )

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self._committed_controls = []

        self._initialize_client(self.rollout_client)
        self._advance_initial_context(self.rollout_client)
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
        predicted_states = self._run_shadow_horizon_rollout(control_plan)

        info: dict[str, Any] = {
            "predicted_states": predicted_states,
            "applied_control": control_plan[0].copy(),
            "control_plan": control_plan,
        }

        reward = self.reward.compute_reward(
            predicted_states=predicted_states,
            control_plan=control_plan,
            weather_forecast=current_obs.weather_forecast,
            energy_price_forecast=current_obs.energy_price_forecast,
            current_obs=current_obs,
            info=info,
        )

        first_control = control_plan[0].copy()
        main_values = self.main_client.advance(
            self._control_row_to_boptest_inputs(first_control)
        ).values

        self._append_to_histories(
            next_weather=current_obs.weather_forecast[0].copy(),
            next_control=first_control,
            next_state=self._extract_state_from_payload(main_values),
        )

        self._committed_controls.append(first_control.copy())
        self.current_time_seconds += self.testcase.step_period_seconds
        self.elapsed_episode_seconds += self.testcase.step_period_seconds

        truncated = self.elapsed_episode_seconds >= self.config.total_simulation_seconds
        return self.get_obs(), float(reward), False, truncated, info

    @jaxtyped(typechecker=beartype)
    def _run_shadow_horizon_rollout(
        self,
        control_plan: Float[np.ndarray, "{self.config.horizon} {self.config.n_controls}"],
    ) -> Float[np.ndarray, "{self.config.horizon} {self.config.n_states}"]:
        self._sync_rollout_to_current_time()

        states: list[FloatArray] = []
        for horizon_step in range(self.config.horizon):
            values = self.rollout_client.advance(
                self._control_row_to_boptest_inputs(control_plan[horizon_step])
            ).values
            states.append(self._extract_state_from_payload(values))

        return np.stack(states, axis=0).astype(np.float64)

    def _sync_rollout_to_current_time(self) -> None:
        if not self._committed_controls:
            return

        self._initialize_client(self.rollout_client)
        self._advance_initial_context(self.rollout_client)
        for committed_control in self._committed_controls:
            self.rollout_client.advance(
                self._control_row_to_boptest_inputs(committed_control)
            )

    def close(self) -> None:
        main_error: Exception | None = None
        try:
            self.main_client.stop()
        except Exception as exc:
            main_error = exc
        finally:
            self.rollout_client.stop()

        if main_error is not None:
            raise main_error

    def __enter__(self) -> "BoptestRolloutEnv":
        return self


__all__ = ["BoptestRolloutEnv"]
