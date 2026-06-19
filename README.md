# hvacbench

[![CI](https://github.com/dcg-udl-cat/hvacbench/actions/workflows/ci.yml/badge.svg)](https://github.com/dcg-udl-cat/hvacbench/actions/workflows/ci.yml)
[![Docs](https://github.com/dcg-udl-cat/hvacbench/actions/workflows/docs.yml/badge.svg)](https://github.com/dcg-udl-cat/hvacbench/actions/workflows/docs.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-2ea44f.svg)](https://dcg-udl-cat.github.io/hvacbench/)

Backend-agnostic environments for building HVAC control research.

`hvacbench` provides a shared receding-horizon environment API for comparing
HVAC control backends. It is designed for workflows where a policy may be
trained with a learned digital twin and evaluated against a physics-based
BOPTEST backend without changing the policy-facing observation and action
contract.

## Why hvacbench?

Building control experiments often bind together controller code, forecast
data, simulator access, reward logic, and evaluation protocol. `hvacbench`
separates those concerns into reusable pieces:

- environments expose a consistent control-plan API;
- providers supply histories, weather forecasts, and electricity prices;
- model wrappers evaluate learned digital twins;
- BOPTEST clients evaluate physics-backed rollouts and realized policy steps;
- reward strategies score comfort, energy cost, and control smoothness.

The current implementation includes a TinyTimeMixer-compatible backend, a
`bestest_air` BOPTEST mapping, mock backends for tests, packaged CSV data
providers, safety filtering, examples, documentation, and CI workflows.

## Environment contract

At every step, a controller receives past histories and future forecasts, then
submits a full `(horizon, 2)` control plan with heating and cooling setpoints in
Celsius. Each backend evaluates the plan according to its semantics and commits
only the first control row before the environment moves forward by one 15-minute
timestep.

The three main environments are:

- `TTMEnv`: learned digital-twin backend using TinyTimeMixer-compatible models.
- `BoptestRolloutEnv`: BOPTEST-backed horizon rollout backend with committed and
  rollout simulators.
- `BoptestEvaluationEnv`: single-simulator BOPTEST backend for realized policy
  evaluation.

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

Minimal learned-digital-twin environment:

```python
from hvacbench.config import EnvConfig, TTMVariables
from hvacbench.envs import TTMEnv
from hvacbench.rewards.simple import SimpleReward

config = EnvConfig()
variables = TTMVariables()
reward = SimpleReward(config=config)

env = TTMEnv(
    config=config,
    reward=reward,
    model_path="gft/ttm4hvac",
    variables=variables,
)

obs, info = env.reset()
control_plan = env.get_random_control_plan()
next_obs, reward_value, terminated, truncated, step_info = env.step(control_plan)
```

For tests or custom integrations, pass an object implementing `BaseTTM` through
the `model` argument instead of passing `model_path`.

## BOPTEST backends

`BoptestRolloutEnv` and `BoptestEvaluationEnv` use the same full-horizon action
shape as `TTMEnv`. By default, they instantiate the packaged `BestestAir`
testcase mapping. You can also pass a custom `BoptestTestcase` implementation.

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
two simultaneous `bestest_air` testcase instances. Use it when a controller
needs BOPTEST-backed horizon rollout scores.

`BoptestEvaluationEnv` uses one BOPTEST simulator, applies only the first
control row, and computes reward from the realized one-step transition. Use it
for final realized policy evaluation.

Run a live rollout smoke test with:

```bash
uv run python examples/run_bestest_air_boptest_rollout_env.py
```

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
tasks are tracked in [PUBLICATION_TODO.md](PUBLICATION_TODO.md).

## License

`hvacbench` is distributed under the MIT license. See [LICENSE.txt](LICENSE.txt).
