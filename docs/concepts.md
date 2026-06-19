# Concepts

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
forecaster or BOPTEST.

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

`BestestAirCsvProvider` uses packaged cyclic CSV data for offline tests and
model-backed rollouts.

## Rewards

Rewards implement `RewardStrategy`. `SimpleReward` combines:

- comfort violation penalty;
- electricity cost;
- setpoint smoothness penalty.

Research users can add domain-specific reward strategies without changing the
environment backends.
