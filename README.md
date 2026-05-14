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
uv sync
```

## Running tests

```sh
uv run pytest
```

## Running example

```sh
uv run python examples/run_mock_env.py
```
