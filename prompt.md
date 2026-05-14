You are an expert Python software engineer. Build the first version of a Python project that implements a Gym-style digital twin environment for training an RL agent to control a building HVAC system.

The goal is to create a clean, maintainable, extensible Python 3.12 project using uv for dependency management.

The project should not be over-engineered, but it should follow good practices: clear structure, Python typing, dataclasses where helpful, protocols/abstract base classes for interfaces, tests, documentation, and simple examples.

Context
=======

We have a trained forecasting model that acts as a digital twin of a building and HVAC system. The model is intended to be used as the world model for a learning RL agent.

At every time step, the RL agent receives past observations plus future forecasts and outputs an entire 96-step trajectory for the control variables.

The digital twin then predicts the next 96 state values. A horizon reward is computed from those predicted states and the proposed controls. Then only the first predicted next state and first control action are actually applied to the environment history. The environment moves forward one timestep and repeats.

This is a receding-horizon setup.

Current variables
=================

The current variables are:

State variables:

    State = [
        "Room Air Temperature (C)",
        "HVAC Power Consumption (W)",
    ]

Weather variables:

    Weather = [
        "Outdoor Air Temperature (C)",
        "Outdoor Humidity (%)",
        "Wind Speed (m/s)",
        "Direct Solar Radiation (W/m^2)",
    ]

Control variables:

    Controls = [
        "Heating Setpoint (C)",
        "Cooling Setpoint (C)",
    ]

However, the implementation should make these easily modifiable. Do not hardcode them deeply throughout the codebase. Use a configuration object or constants module so the variable names and dimensions can be changed easily.

Main abstractions
=================

Implement the following conceptual classes/interfaces:

    Env
        - get_obs()
        - step(action)

    TTMEnv: Env
        - Environment backed by the trained forecasting model / digital twin.
        - Constructor:
            TTMEnv(
                provider: FutureDataProvider,
                reward: RewardStrategy,
                model: TTMSurrogateModel,
                ...
            )

    SafeEnv: Env
        - Decorator around another Env.
        - Applies a safety layer to the agent-proposed control trajectory before passing it to the wrapped environment.
        - The wrapped environment should know nothing about the safety layer.

    BoptestEnv: Env
        - Stub class only for now.
        - It will eventually be another Env implementation backed by a building simulator instead of a forecasting model.
        - It should be transparent to code that consumes an object of type Env.
        - Do not implement BOPTEST integration yet. Raise NotImplementedError with clear messages.

    RewardStrategy
        - compute_reward(...)

    SimpleReward: RewardStrategy
        - Start with a simple horizon reward.

    FutureDataProvider
        - get_energy_price_forecast(...)
        - get_weather_forecast(...)
        - It should also provide or support initial history data for reset/startup.

    MockProvider: FutureDataProvider
        - Simple mock implementation.
        - Produces deterministic or pseudo-random weather forecasts and energy prices.
        - Provides initial histories for weather, controls, and states.

    TTMSurrogateModel
        - Interface/protocol for the trained digital twin model.
        - The actual trained model is not available yet.
        - Provide a mock implementation for testing and demonstration.

Digital twin model
==================

The real model expects:

Inputs:

    - Last 1536 observations of weather variables.
    - Last 1536 observations of control variables.
    - Last 1536 observations of state variables.
    - Next 96 observations of weather variables.
    - Next 96 planned values for control variables.

Outputs:

    - Next 96 values for state variables.

Use the following default dimensions:

    history_length = 1536
    horizon = 96
    n_weather = 4
    n_controls = 2
    n_states = 2

The default expected array shapes should be:

    weather_history.shape == (1536, 4)
    control_history.shape == (1536, 2)
    state_history.shape == (1536, 2)
    weather_forecast.shape == (96, 4)
    control_plan.shape == (96, 2)
    predicted_states.shape == (96, 2)

The implementation should validate shapes and raise helpful errors.

Receding horizon logic
======================

At time t:

1. TTMEnv has internal histories:
       weather_history
       control_history
       state_history

2. TTMEnv asks the provider for:
       weather_forecast = provider.get_weather_forecast(t, horizon)
       energy_price_forecast = provider.get_energy_price_forecast(t, horizon)

3. TTMEnv exposes an observation containing:
       weather_history
       control_history
       state_history
       weather_forecast
       energy_price_forecast

4. Agent passes a full control trajectory:
       control_plan.shape == (96, n_controls)

5. TTMEnv calls:
       predicted_states = model.predict(
           weather_history=...,
           control_history=...,
           state_history=...,
           weather_forecast=...,
           control_plan=...,
       )

6. TTMEnv computes a horizon reward:
       reward = reward_strategy.compute_reward(
           predicted_states=predicted_states,
           control_plan=control_plan,
           weather_forecast=weather_forecast,
           energy_price_forecast=energy_price_forecast,
           current_obs=obs,
           info=optional_info,
       )

7. Receding horizon application:
       applied_control = control_plan[0]
       next_state = predicted_states[0]
       next_weather = weather_forecast[0]

8. TTMEnv appends these to its histories and drops the oldest observations:
       control_history <- append applied_control
       state_history <- append next_state
       weather_history <- append next_weather

9. TTMEnv increments internal timestep t by 1.

10. TTMEnv returns:
       next_obs, reward, terminated, truncated, info

Use a Gymnasium-like return signature from step():

    tuple[Observation, float, bool, bool, dict[str, Any]]

Where:
    terminated = False by default for now
    truncated = True only if the provider has a finite data source and reaches the end
    info contains useful debugging fields:
        - predicted_states
        - applied_control
        - control_plan
        - safe_control_plan if applicable
        - reward_components if available

Observation representation
==========================

Create a typed dataclass for observations, for example:

    @dataclass(frozen=True)
    class Observation:
        weather_history: NDArray[np.float64]
        control_history: NDArray[np.float64]
        state_history: NDArray[np.float64]
        weather_forecast: NDArray[np.float64]
        energy_price_forecast: NDArray[np.float64] | None = None

It should be easy for an RL library to flatten this later, but for now keep it structured and readable.

Configuration
=============

Create a config dataclass, for example:

    EnvConfig:
        history_length: int = 1536
        horizon: int = 96
        state_variables: tuple[str, ...] = (...)
        weather_variables: tuple[str, ...] = (...)
        control_variables: tuple[str, ...] = (...)

Add computed properties:
    n_states
    n_weather
    n_controls

Use this config throughout the project.

Reward
======

Implement SimpleReward.

Start with a simple horizon reward that penalizes:

1. Comfort violations based on indoor room temperature.
2. Energy cost based on predicted HVAC power consumption and energy price.
3. Optional control smoothness.

Assume:

    state variable 0 = room air temperature in Celsius
    state variable 1 = HVAC power consumption in W

Default comfort bounds:

    temp_min = 20.0
    temp_max = 24.0

Energy cost:

    HVAC Power Consumption (W) should be converted to kW:
        power_kw = power_w / 1000

If the timestep duration is needed, include it in config or reward parameters. Use default:

    timestep_minutes = 15

Then energy in kWh per step:

    energy_kwh = power_kw * timestep_hours

Energy price forecast can initially be a simple array of length 96:

    energy_price_forecast.shape == (96,)

where each value is something like EUR/kWh.

Reward should be negative cost:

    reward = - (
        comfort_weight * comfort_penalty
        + energy_weight * energy_cost
        + smoothness_weight * smoothness_penalty
    )

Also return reward components in info if convenient. Do not make the API too complex, but make debugging possible.

Safety layer
============

Implement SafeEnv as a decorator around another Env.

Example:

    base_env = TTMEnv(provider=provider, reward=reward, model=model)
    env = SafeEnv(base_env, safety_filter=safety_filter)

The TTMEnv must not know about the safety layer.

The safety layer should:

1. Validate control trajectory shape.
2. Clip controls to configured min/max limits.
3. Optionally enforce heating setpoint <= cooling setpoint.
4. Optionally enforce maximum ramp rate between consecutive control values.

Default safety constraints for the current two controls:

    Heating Setpoint (C):
        min = 15.0
        max = 24.0

    Cooling Setpoint (C):
        min = 22.0
        max = 30.0

    heating_setpoint <= cooling_setpoint

If heating_setpoint > cooling_setpoint, repair the action in a simple deterministic way. For example, set both around their midpoint while respecting bounds, or set cooling = max(cooling, heating). Choose a simple and clear approach and document it.

Implementation suggestion:

    ControlSafetyFilter
        - apply(control_plan, previous_control=None) -> safe_control_plan

Then SafeEnv.step(action) should:
    - transform action into safe_action
    - pass safe_action to wrapped_env.step(safe_action)
    - add original_action and safe_action to info

Mock model
==========

Implement a MockTTMSurrogateModel for tests and examples.

It should produce plausible predicted states with correct shape, not necessarily physically perfect.

Example behavior:
    - Indoor temperature slowly moves toward the average of heating/cooling setpoints and outdoor temperature.
    - HVAC power increases when the setpoints require conditioning.
    - Make it deterministic by default.

The mock model should be simple and readable.

Mock provider
=============

Implement MockProvider.

It should:
    - Provide initial histories for weather, controls, and states.
    - Provide weather forecast for any timestep.
    - Provide energy price forecast for any timestep.

Make it deterministic by default using a seed.

Weather forecast can use simple sinusoidal daily temperature patterns and simple constant or pseudo-random values for humidity, wind, and solar radiation.

Energy price can be constant or have a simple daily peak/off-peak pattern.

Project structure
=================

Use a clean project layout. Suggested structure:

    pyproject.toml
    README.md
    src/
      hvacbench/
        __init__.py
        config.py
        types.py
        protocols.py
        envs/
          __init__.py
          base.py
          ttm_env.py
          safe_env.py
          boptest_env.py
        rewards/
          __init__.py
          base.py
          simple.py
        providers/
          __init__.py
          base.py
          mock.py
        models/
          __init__.py
          base.py
          mock.py
        safety/
          __init__.py
          control_safety.py
        utils/
          __init__.py
          validation.py
    tests/
      test_ttm_env.py
      test_safe_env.py
      test_simple_reward.py
      test_mock_provider.py
      test_mock_model.py
    examples/
      run_mock_env.py

Dependencies
============

Use uv.

Use Python 3.12.

Keep runtime dependencies minimal:

    numpy

Development dependencies:

    pytest
    ruff
    mypy

Optional:
    gymnasium

Do not require gymnasium unless you decide it is actually useful. The project should work without needing to inherit from gymnasium.Env. We only need a Gym-style API for now.

Create pyproject.toml configured for uv.

Include useful scripts or documented commands:

    uv sync
    uv run pytest
    uv run ruff check .
    uv run mypy src

Typing
======

Use modern Python typing:

    from __future__ import annotations
    typing.Protocol
    typing.Any
    numpy.typing.NDArray

Use type aliases where helpful:

    FloatArray = NDArray[np.float64]

Avoid excessive abstraction, but make interfaces clear.

Testing
=======

Add tests that verify:

1. TTMEnv.reset or initialization creates valid observations.
2. TTMEnv.step accepts a valid control plan and returns:
       next_obs, reward, terminated, truncated, info
3. Histories stay at length 1536 after stepping.
4. The first predicted state is appended to state_history.
5. The first weather forecast row is appended to weather_history.
6. The first control action is appended to control_history.
7. Shape validation fails with helpful errors for invalid action shapes.
8. SimpleReward returns a finite float.
9. SafeEnv clips unsafe actions.
10. SafeEnv ensures heating_setpoint <= cooling_setpoint.
11. BoptestEnv exists but raises NotImplementedError for get_obs and step.

README
======

Write a README explaining:

1. What the project is.
2. The receding-horizon environment logic.
3. The main abstractions:
       Env
       TTMEnv
       SafeEnv
       BoptestEnv
       RewardStrategy
       FutureDataProvider
       TTMSurrogateModel
4. How to install dependencies using uv.
5. How to run tests.
6. How to run the example.
7. How to replace MockTTMSurrogateModel with a real trained model.
8. How to modify variables and dimensions through EnvConfig.

Example
=======

Create examples/run_mock_env.py that:

1. Creates EnvConfig.
2. Creates MockProvider.
3. Creates MockTTMSurrogateModel.
4. Creates SimpleReward.
5. Creates TTMEnv.
6. Wraps it with SafeEnv.
7. Runs 10 environment steps using a simple constant 96-step control plan.
8. Prints:
       step number
       reward
       latest room temperature
       latest HVAC power
       applied control

Quality expectations
====================

The code should be:

    - Simple
    - Readable
    - Typed
    - Tested
    - Easy to extend
    - Not over-engineered

Avoid:
    - Large frameworks
    - Complex dependency injection systems
    - Premature optimization
    - Hidden global state
    - Hardcoded dimensions scattered across files

Important design decisions
==========================

1. TTMEnv should not implement safety logic.
   Safety belongs in SafeEnv / safety filters.

2. BoptestEnv should implement the same Env interface eventually, but for now it is a stub.

3. FutureDataProvider should isolate all future exogenous data:
       weather forecast
       energy price forecast
       initial histories

4. TTMSurrogateModel should isolate the trained model.
   The environment should not care whether the model is PyTorch, ONNX, TensorFlow, scikit-learn, or a mock.

5. The environment should use receding-horizon logic:
       action is a full 96-step trajectory
       prediction is a full 96-step state trajectory
       reward is computed over the full horizon
       only the first control and first predicted next state are applied

6. Use 1536 for weather, control, and state history length.

Deliverables
============

Please generate the complete project files.

At the end, provide:

1. A concise summary of what was implemented.
2. The full file tree.
3. The commands to run:
       uv sync
       uv run pytest
       uv run python examples/run_mock_env.py
4. Any assumptions made.