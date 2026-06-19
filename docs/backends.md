# Backends

## TTMEnv

`TTMEnv` evaluates horizon control plans with a TinyTimeMixer-compatible
forecasting model. It is intended for training receding-horizon control policies
on learned building surrogates derived from operation data.

The model is expected to receive recent weather, control, and state histories
plus future weather and control trajectories. It returns forecasted building
states over the prediction horizon. `TTMEnv` scores that horizon, commits only
the first control and first predicted state, and advances the histories by one
timestep.

A caller can provide either:

- `model_path`, which constructs the packaged `TTM` wrapper; or
- `model`, an object implementing `BaseTTM`.

The environment may keep a longer internal history buffer than the observation
window if the model context length is larger than `EnvConfig.history_length`.
The packaged default provider uses BOPTEST `bestest_air` operation data.

## BoptestRolloutEnv

`BoptestRolloutEnv` is the physics-based counterpart to `TTMEnv`. It accepts the
same full-horizon control plan, but obtains the resulting state trajectory by
rolling out BOPTEST instead of asking a learned forecaster.

It uses two BOPTEST testcase instances:

- a committed simulator for applying the first selected control;
- a rollout simulator for evaluating the candidate horizon plan.

Before each horizon rollout, the rollout simulator is synchronized by replaying
the committed control history. This provides the same receding-horizon semantics
as `TTMEnv` while treating BOPTEST as the trusted reference dynamics.

## BoptestEvaluationEnv

`BoptestEvaluationEnv` uses one BOPTEST testcase instance. It accepts the same
full-horizon control plan as the other environments, applies only the first
control row, and computes the reward from the realized one-step transition.

This backend is intended for final policy evaluation when shadow horizon access
to BOPTEST should not be part of the scoring protocol. It is closest to a real
deployment setting: the policy may propose a horizon, but only the first action
can be committed before observing the next state.

## bestest_air assumptions

- Heating and cooling setpoints are represented internally in Celsius.
- BOPTEST setpoint inputs are sent in Kelvin.
- Room air temperature is converted from Kelvin to Celsius.
- HVAC power is approximated as cooling power plus fan power plus heating power.
- Electricity price point selection follows `EnergyPriceType`.

## Future backend paths

`bestest_air` is the current default because it provides a concrete and
repeatable validation target. The environment interfaces are designed so future
work can add more BOPTEST testcases, other simulator-backed environments such as
Sinergym, and other forecasting model wrappers beyond TinyTimeMixer.
