# Getting started

## Requirements

- Python 3.13
- `uv`

The package pins Python to `>=3.13,<3.14` because the current forecasting-model
dependency stack is validated on Python 3.13.

## Install from source

```bash
git clone https://github.com/dcg-udl-cat/hvacbench.git
cd hvacbench
uv sync --all-extras
```

## Run the test suite

```bash
uv run pytest
uv run ruff check src tests examples
```

## Run a local smoke test

```bash
uv run python examples/run_mock_env.py
```

The mock example does not require a BOPTEST server or a trained TinyTimeMixer
checkpoint. It uses deterministic mock provider and model implementations to
exercise the environment loop.

## Minimal learned-surrogate environment

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

The default example uses
[`gft/ttm4hvac`](https://huggingface.co/gft/ttm4hvac), a TinyTimeMixer HVAC
fine-tune published on Hugging Face. You can also use your own fine-tuned TTM
checkpoint, or pass a custom model object implementing `BaseTTM`.

## Minimal BOPTEST rollout environment

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
)
```

`BoptestRolloutEnv` requires a running BOPTEST service with enough workers for
two simultaneous `bestest_air` testcase instances. By default it creates the
packaged `BestestAir` testcase mapping; pass `testcase=...` to use another
`BoptestTestcase` implementation.

Use `BoptestEvaluationEnv` for deployment-style evaluation: it accepts the same
control plan shape, applies only the first action, and computes reward from the
realized one-step transition.
