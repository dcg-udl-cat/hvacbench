# hvacbench

`hvacbench` is a Python library for validating data-driven building surrogate
models in receding-horizon HVAC control workflows. It makes it straightforward
to train a policy in an environment backed by a learned forecasting model and
then evaluate the same policy in BOPTEST-backed environments treated as trusted
physics-based references.

## Why it exists

Physics-based building models are costly to construct when detailed thermal,
HVAC, and control components must be represented. Learned surrogates built from
operation data are easier to obtain, but they can be unreliable, particularly
when the model is a black-box deep learning forecaster. `hvacbench` focuses on
that gap:

- `TTMEnv` turns a fine-tuned forecasting model into a control-training
  environment;
- BOPTEST-backed environments expose the same policy-facing contract;
- policies can be moved from learned surrogate training to physics-based
  evaluation;
- rewards receive the proposed controls, future states, weather, and energy
  prices;
- providers supply histories, weather forecasts, and price forecasts;
- the receding-horizon semantics are explicit and shared.

## Current backends

- `TTMEnv`: learned-surrogate backend using TinyTimeMixer-compatible models.
- `BoptestRolloutEnv`: two-simulator BOPTEST backend for horizon rollout
  evaluation.
- `BoptestEvaluationEnv`: single-simulator BOPTEST backend for realized policy
  evaluation.

The current concrete mapping targets the BOPTEST `bestest_air` testcase. That
specific implementation is also the first validation target for TTM4HVAC-style
surrogates. The interfaces are intentionally small so that additional
forecasting models, BOPTEST testcases, reward functions, and future
simulator-backed environments can be added without changing controller code.

## TTM4HVAC starting point

The default model examples use
[`gft/ttm4hvac`](https://huggingface.co/gft/ttm4hvac), a Hugging Face repository
for TinyTimeMixer HVAC dynamics modeling. The project can also use custom
fine-tunes or other forecasting models adapted to the `BaseTTM` interface.

## Project status

`hvacbench` is research software prepared for open-source publication and a
SoftwareX submission. The package is currently pre-1.0 and the public API may
change while the surrogate-validation workflow matures.
