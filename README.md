# hvacbench

[![CI](https://github.com/dcg-udl-cat/hvacbench/actions/workflows/ci.yml/badge.svg)](https://github.com/dcg-udl-cat/hvacbench/actions/workflows/ci.yml)
[![Docs](https://github.com/dcg-udl-cat/hvacbench/actions/workflows/docs.yml/badge.svg)](https://github.com/dcg-udl-cat/hvacbench/actions/workflows/docs.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-2ea44f.svg)](https://dcg-udl-cat.github.io/hvacbench/)

Receding-horizon environments for validating learned HVAC surrogate models.

`hvacbench` helps evaluate whether data-driven building surrogates are useful
for training HVAC control policies. The core workflow is to train a policy in a
Gym-like environment backed by a learned forecasting model, then evaluate the
same policy in BOPTEST-backed environments that act as trusted physics-based
references for the building.

## Purpose

High-fidelity physics models for HVAC control are expensive to build and
maintain. Surrogates learned from building operation data are easier to obtain,
but they are harder to trust, especially when they are based on black-box deep
learning forecasters.

`hvacbench` is built around that validation problem. It gives learned
surrogates and physics-based BOPTEST simulations the same receding-horizon
control interface, so a controller trained on a data-driven model can be tested
against a more reliable simulator without changing policy code.

## Validation workflow

1. Fine-tune or provide a forecasting model that predicts future building states from histories, weather forecasts, and a proposed future control plan. If none is available, we encourage using those in [https://huggingface.co/gft/tm4hvac](htttps://huggingface.co/gft/ttm4hvac) with the `TTMEnv` environment.
2. Train a receding-horizon control policy with this env, which uses that model
   as the environment dynamics. By default, the env uses data from the BOPTEST `bestest_air` testcase, which makes it easy to compare the learned surrogate with the BOPTEST reference.
3. Evaluate the resulting policy with `BoptestRolloutEnv`, where BOPTEST rolls
   out the same proposed control horizon using two synchronized simulator
   clients.
4. Evaluate deployment-like behavior with `BoptestEvaluationEnv`, where only the
   first proposed action is applied, as would happen on a real building.

This setup makes the learned surrogate itself easier to validate: if policies
trained on it transfer well to the BOPTEST-backed environments, the surrogate is
more credible for control-oriented use.

## Implemented environments

- `TTMEnv`: a learned-surrogate environment for TinyTimeMixer-compatible
  forecasting models. The model receives histories and a future control plan,
  predicts future states over the horizon, and the environment commits only the
  first predicted transition.
- `BoptestRolloutEnv`: a BOPTEST-backed horizon rollout environment. It uses one
  committed simulator and one rollout simulator to evaluate the full proposed
  control plan, then commits only the first action.
- `BoptestEvaluationEnv`: a deployment-style BOPTEST environment. It accepts the
  same full control plan but applies only the first row.

All three environments use the same `(horizon, 2)` control-plan contract for
heating and cooling setpoints in Celsius. By default, the provided TTM data
provider and BOPTEST mappings target the BOPTEST `bestest_air` testcase.

## Rewards

Environments receive a reward object implementing `RewardStrategy`. A reward is
computed from the proposed control plan, forecasted or realized future states,
weather observations, and energy-price forecasts. The packaged `SimpleReward`
combines comfort, energy cost, and setpoint smoothness terms, but experiments can
provide their own reward strategy.

## Installation

The project currently targets Python `>=3.13,<3.14` and uses `uv` for local
development.

```bash
git clone https://github.com/dcg-udl-cat/hvacbench.git
cd hvacbench
uv sync --all-extras --group dev
```

Once the package is released, PyPI installation instructions will be added here.

## Quick start

Run the deterministic mock environment without a trained model or BOPTEST
server:

```bash
uv run python examples/run_mock_env.py
```

Minimal learned-surrogate environment:

```python
from hvacbench.config import EnvConfig, TTMVariables
from hvacbench.envs import TTMEnv
from hvacbench.rewards.simple import SimpleReward

config = EnvConfig()
reward = SimpleReward(config=config)

env = TTMEnv(
    config=config,
    reward=reward,
    model_path="gft/ttm4hvac",
)

obs, info = env.reset()
control_plan = env.get_random_control_plan()
next_obs, reward_value, terminated, truncated, step_info = env.step(control_plan)
```

The current first-step model path is the
[`gft/ttm4hvac`](https://huggingface.co/gft/ttm4hvac) Hugging Face repository,
which hosts a TinyTimeMixer HVAC fine-tune and related datasets/models. You can
also pass a custom object implementing `BaseTTM` through the `model` argument.

## BOPTEST evaluation

`BoptestRolloutEnv` and `BoptestEvaluationEnv` use `BestestAir` by default. You
can also pass another `BoptestTestcase` implementation.

```python
from hvacbench.config import EnvConfig
from hvacbench.energy_price import EnergyPriceType
from hvacbench.envs import BoptestRolloutEnv
from hvacbench.rewards.simple import SimpleReward

config = EnvConfig(history_length=8, horizon=8)
reward = SimpleReward(config=config)

env = BoptestRolloutEnv(
    config=config,
    reward=reward,
    energy_price_type=EnergyPriceType.DYNAMIC,
    start_day=0,
)
```

`BoptestRolloutEnv` requires a running BOPTEST service with enough workers for
two simultaneous `bestest_air` testcase instances.

Run a live rollout smoke test with:

```bash
uv run python examples/run_bestest_air_boptest_rollout_env.py
```

## Extensibility

The current implementation provides the TTM path first because TinyTimeMixer is
a practical foundation time-series model for HVAC dynamics. The architecture is
kept deliberately small so other forecasting models, BOPTEST testcases, reward
functions, and simulator backed environments such as Sinergym based
backends can be added without rewriting controller code.

## Documentation

The documentation site is configured for GitHub Pages:

https://dcg-udl-cat.github.io/hvacbench/

Build it locally with:

```bash
uv run mkdocs build --strict
```

## Development checks

```bash
uv lock --check
uv run ruff check src tests examples
uv run pytest
uv build
```

## Publication status

`hvacbench` is pre-1.0 research software being prepared for public release,
PyPI packaging, and a SoftwareX submission. Remaining release and manuscript
tasks are tracked in [PUBLICATION_TODO.md](TODO.md).

## License

`hvacbench` is distributed under the MIT license. See [LICENSE.txt](LICENSE.txt).
