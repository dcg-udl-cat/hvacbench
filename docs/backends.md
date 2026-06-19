# Backends

## TTMEnv

`TTMEnv` evaluates horizon control plans with a TinyTimeMixer-compatible model.
It is intended for fast learned-digital-twin training or screening. A caller can
provide either:

- `model_path`, which constructs the packaged `TTM` wrapper; or
- `model`, an object implementing `BaseTTM`.

The environment may keep a longer internal history buffer than the observation
window if the model context length is larger than `EnvConfig.history_length`.

## BoptestRolloutEnv

`BoptestRolloutEnv` uses two BOPTEST testcase instances:

- a committed simulator for applying the first selected control;
- a rollout simulator for evaluating the candidate horizon plan.

Before each horizon rollout, the rollout simulator is synchronized by replaying
the committed control history. This prioritizes correct receding-horizon
semantics over speed.

## BoptestEvaluationEnv

`BoptestEvaluationEnv` uses one BOPTEST testcase instance. It accepts the same
full-horizon control plan as the other environments, applies only the first
control row, and computes the reward from the realized one-step transition.

This backend is intended for final policy evaluation when shadow horizon access
to BOPTEST should not be part of the scoring protocol.

## bestest_air assumptions

- Heating and cooling setpoints are represented internally in Celsius.
- BOPTEST setpoint inputs are sent in Kelvin.
- Room air temperature is converted from Kelvin to Celsius.
- HVAC power is approximated as cooling power plus fan power plus heating power.
- Electricity price point selection follows `EnergyPriceType`.
