# Concepts

## Surrogate validation workflow

The main workflow is:

1. provide a forecasting model trained on building operation data;
2. train a policy with `TTMEnv`, where that model supplies the dynamics;
3. evaluate the same policy with BOPTEST-backed environments;
4. compare whether policy behavior learned on the surrogate remains valid under
   the physics-based reference.

This is a control-oriented validation loop for learned building surrogates. It
does not require the surrogate to be an interpretable physics model, but it does
test whether a policy trained with the surrogate behaves plausibly when deployed
against a more trusted simulator.

## Receding-horizon action contract

All environments accept a full control trajectory of shape
`(horizon, n_controls)`. For the current configuration, controls are heating and
cooling setpoints in Celsius.

At each environment step:

1. The agent receives histories and forecasts.
2. The agent proposes a full horizon control plan.
3. The backend evaluates the proposed plan.
4. The reward is computed.
5. Only the first control and first resulting state are committed.
6. Histories advance by one timestep.

This contract keeps policy code independent of whether the backend is a learned
forecaster, a BOPTEST rollout environment, or a deployment-style BOPTEST
evaluation environment.

## Observation

`Observation` contains:

- weather history;
- control history;
- state history;
- weather forecast;
- energy price forecast.

The state vector currently contains room air temperature and HVAC power. The
weather vector contains outdoor temperature, outdoor humidity, wind speed, and
direct solar radiation.

## Providers

Providers implement `BaseProvider` and are responsible for:

- initial history windows;
- weather forecasts;
- electricity price forecasts;
- random control-plan sampling.

`BestestAirCsvProvider` uses packaged cyclic CSV data generated from BOPTEST
`bestest_air` operation for offline tests and model-backed rollouts.

## Rewards

Rewards implement `RewardStrategy`. `SimpleReward` combines:

- comfort violation penalty;
- electricity cost;
- setpoint smoothness penalty.

The reward object receives the proposed control plan, the predicted or realized
future states, weather observations, energy-price forecasts, the current
observation, and a mutable `info` dictionary for diagnostics. Research users can
add domain-specific reward strategies without changing the environment
backends.
