# hvacbench

A Gym-style digital twin environment for reinforcement learning control of building HVAC systems.

## Overview
This project implements an RL environment backed by a forecasting model structure acting as a digital twin. It is organized around receding-horizon control logic, meaning the agent provides a full action trajectory for the planning horizon instead of merely a single timestep action.

## Core Abstractions

- **`BaseEnv`**: The environment protocol.
- **`TTMEnv`**: Provides the simulation loop driven by historical context, future forecasts, and a TinyTimeMixer-compatible model.
- **`SafeEnv`**: A decorator for safe interactions wrapper. Verifies or adjusts out-of-bounds agent directives.
- **`BaseProvider`**: Interface for history initialization plus weather and price forecasts. `BestestAirCsvProvider` is the packaged CSV-backed implementation.
- **`RewardStrategy`**: Exposes flexible logic for optimization goals like comfort range keeping, energy expenditure savings, and system smooth operations calculations. 
- **`BaseTTM` / `TTM`**: Model interfaces for predicting state trajectories from histories, forecasts, and proposed controls.
- **`BoptestRolloutEnv`**: A two-simulator BOPTEST backend for horizon rollout scoring.
- **`BoptestEvaluationEnv`**: A single-simulator BOPTEST backend for realized policy evaluation.

This repository contains the reusable library code only. Experiment orchestration
and training scripts should live in a separate project that imports `hvacbench`.

## Dependencies

You need Python `3.13` and `uv`.

## Installation

```sh
uv sync --all-extras
```

## Running tests

```sh
uv run pytest
```

## Running example

```sh
uv run python examples/run_mock_env.py
```

## Configuration

`EnvConfig` only contains environment sizing and episode length:

```python
from hvacbench.config import EnvConfig

config = EnvConfig(
    history_length=1536,
    horizon=96,
    total_simulation_seconds=14 * 24 * 3600,
)
```

TTM model column names are supplied separately:

```python
from hvacbench.config import TTMVariables

variables = TTMVariables()
```

Pass the same `variables` object to `TTMEnv` and to the real `TTM` model so the
model preprocessor validation checks the expected state, weather, and control
columns.

`TTMEnv` can construct the packaged `TTM` wrapper from a model path:

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
```

For tests or custom integrations, pass an object implementing `BaseTTM` through
the `model` argument. In that case `model_path` is not required.

Create the packaged `bestest_air` CSV provider with one of the available price
scenarios:

```python
from hvacbench.energy_price import EnergyPriceType
from hvacbench.providers import BestestAirCsvProvider

provider = BestestAirCsvProvider(
    config=config,
    energy_price_type=EnergyPriceType.DYNAMIC,
    variables=variables,
)
```

The provider treats the annual data as cyclic, so forecasts and histories wrap
across the end of the year instead of truncating the data source. By default it
uses the packaged repository CSVs; pass `building_data_path` and
`electricity_price_data_path` to load compatible CSVs from another location.

`TTMEnv(start_day=0)` starts provider lookups at day 0 by default. Pass a
different non-negative day to offset both initial histories and forecasts into
the provider data without changing the configured episode length.

## BOPTEST bestest_air Environments

Both BOPTEST environments use the same action contract as `TTMEnv`: the agent
submits a full `(horizon, 2)` control plan in internal Celsius setpoints.

`BoptestRolloutEnv` uses two independent BOPTEST `bestest_air` testcase
instances:

- `main_client` is the committed simulator.
- `rollout_client` is synchronized by replaying the committed first-step
  controls before evaluating the candidate 96-step control plan.

Use it when you need BOPTEST-backed receding-horizon rollout scores.

```python
import numpy as np

from hvacbench.config import EnvConfig
from hvacbench.boptest.bestest_air import BestestAir
from hvacbench.energy_price import EnergyPriceType
from hvacbench.envs import BoptestRolloutEnv
from hvacbench.rewards.simple import SimpleReward

config = EnvConfig()
testcase = BestestAir(
    base_url="http://127.0.0.1",
    energy_price_type=EnergyPriceType.DYNAMIC,
)
reward = SimpleReward(config=config)

env = BoptestRolloutEnv(
    reward=reward,
    config=config,
    testcase=testcase,
    start_day=0,
)

control_plan = np.tile(
    np.array([[21.0, 24.0]], dtype=np.float64),
    (config.horizon, 1),
)
next_obs, reward_value, terminated, truncated, step_info = env.step(control_plan)
```

The shadow simulator is synchronized by reinitializing the rollout testcase,
advancing through the fixed initial context, and replaying all committed
first-step controls before the 96-step candidate rollout. This prioritizes
correct receding-horizon semantics over speed.

`BoptestEvaluationEnv` is the evaluation-oriented BOPTEST backend. It accepts
the same full-horizon action shape, but uses one BOPTEST simulator, applies only
the first control row, and computes reward from the realized one-step
transition. Use it for final realized policy evaluation.

Run the live BOPTEST rollout example with:

```sh
uv run python examples/run_bestest_air_boptest_rollout_env.py
```

It requires a running BOPTEST service with enough workers for two simultaneous
`bestest_air` testcase instances. The example uses a small `history_length=8`
and `horizon=8` smoke-test configuration so it finishes quickly; the default
research configuration remains `history_length=1536` and `horizon=96`.

### bestest_air Assumptions

- Only the `bestest_air` testcase is supported.
- Heating and cooling setpoints are internal Celsius values and are sent to
  BOPTEST as Kelvin.
- Room air temperature is read from Kelvin and converted back to Celsius.
- HVAC power is approximated as
  `fcu_reaPCoo_y + fcu_reaPFan_y + fcu_reaPHea_y`. `fcu_reaPHea_y` is thermal
  heating power, not necessarily electrical power, so this is a pragmatic first
  signal for the existing internal reward.
- The electricity price forecast point is selected from
  `PriceElectricPowerConstant`, `PriceElectricPowerDynamic`, or
  `PriceElectricPowerHighlyDynamic` based on `BestestAir.energy_price_type`.
- `start_day` is converted to BOPTEST `start_time` seconds during
  `/initialize`; after that, the environment advances the configured initial
  history context before the first control step.
- BOPTEST `/forecast` returns `horizon / interval + 1` rows, so the environment
  requests `(steps - 1) * 900` seconds to receive exactly `steps` rows.
- BOPTEST `/results` is used only for measurement points. Forecast-only weather
  points such as `TDryBul` are read through `/forecast`, including the initial
  weather history window captured before the context warmup is advanced.
- Initial controls are filled with the default setpoints `[21.0, 24.0]`.
