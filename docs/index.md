# hvacbench

`hvacbench` is a Python library for comparing building HVAC control backends
through a shared receding-horizon environment API. It is designed for research
workflows where a controller may be trained with a learned digital twin and then
evaluated against a physics-based backend such as BOPTEST.

## Why it exists

Building control papers often couple a policy, a simulator, a dataset, and a
reward into one experiment script. That makes it hard to compare whether a
result comes from the controller itself or from the backend used to score it.
`hvacbench` separates these concerns:

- environments expose the same observation and control-plan contract;
- providers supply histories, weather forecasts, and price forecasts;
- models predict state trajectories for learned-digital-twin rollouts;
- rewards score comfort, cost, and control regularity over a horizon.

## Current backends

- `TTMEnv`: learned digital-twin backend using TinyTimeMixer-compatible models.
- `BoptestRolloutEnv`: two-simulator BOPTEST backend for horizon rollout scoring.
- `BoptestEvaluationEnv`: single-simulator BOPTEST backend for realized policy
  evaluation.

The current BOPTEST mapping targets the `bestest_air` testcase. The public
interfaces are intentionally small so that additional buildings, providers,
models, and reward functions can be added without changing controller code.

## Project status

`hvacbench` is research software prepared for open-source publication and a
SoftwareX submission. The package is currently pre-1.0 and the public API may
change while the benchmark interface matures.
