# TTM RL Environment

A Gym-style digital twin environment for reinforcement learning control of building HVAC systems.

## Overview
This project implements an RL environment backed by a forecasting model structure acting as a digital twin. It is organized around receding-horizon control logic, meaning the agent provides a full action trajectory for the planning horizon instead of merely a single timestep action.

## Core Abstractions

- **`Env`**: The environment protocol.
- **`TTMEnv`**: Provides the simulation loop driven by historic contexts, future predictions, and surrogate performance models (RL view).
- **`SafeEnv`**: A decorator for safe interactions wrapper. Verifies or adjusts out-of-bounds agent directives.
- **`FutureDataProvider`**: Handles historical arrays initialization and forecast information like weather and prices.
- **`RewardStrategy`**: Exposes flexible logic for optimization goals like comfort range keeping, energy expenditure savings, and system smooth operations calculations. 
- **`TTMSurrogateModel`**: Model interface simulating dynamics of the space predicting ensuing outputs based on combined state controls array history and the incoming forecast properties.
- **`BoptestEnv`**: A template mock connecting standard dynamic systems testing platform later avoiding core change demands.

## Dependencies

You need python `3.12` and `uv`.

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

## BOPTEST bestest_air Environment

`BoptestEnv` implements the same receding-horizon control contract as `TTMEnv`,
but uses two independent BOPTEST `bestest_air` testcase instances:

- `main_client` is the committed simulator.
- `rollout_client` is synchronized by replaying the committed first-step
  controls before evaluating the candidate 96-step control plan.

The agent action shape is `(96, 2)` in internal Celsius setpoints:

```python
import numpy as np

from hvacbench.config import EnvConfig
from hvacbench.boptest.bestest_air import BestestAir
from hvacbench.envs.boptest_env import BoptestEnv
from hvacbench.rewards.simple import SimpleReward

config = EnvConfig()
testcase = BestestAir(
    base_url="http://127.0.0.1",
    energy_price_type="dynamic",  # "constant", "dynamic", or "highly_dynamic"
)
reward = SimpleReward(config=config)

env = BoptestEnv(
    reward=reward,
    config=config,
    testcase=testcase,
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

Run the live BOPTEST example with:

```sh
uv run python examples/run_bestest_air_boptest_env.py
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
- BOPTEST `/forecast` returns `horizon / interval + 1` rows, so the environment
  requests `(steps - 1) * 900` seconds to receive exactly `steps` rows.
- BOPTEST `/results` is used only for measurement points. Forecast-only weather
  points such as `TDryBul` are read through `/forecast`, including the initial
  weather history window captured before the context warmup is advanced.
- Initial controls are filled with the default setpoints `[21.0, 24.0]`.
