# Development

## Local setup

```bash
uv sync --all-extras
```

## Checks

```bash
uv lock --check
uv run pytest
uv run ruff check src tests examples
uv run mkdocs build --strict
uv build
```

Remove generated `dist/` files before committing unless preparing an actual
release artifact.

## Repository layout

```text
src/hvacbench/      Python package
tests/              unit tests and backend contract tests
examples/           executable examples
docs/               GitHub Pages documentation
```

## Adding a backend

New backends should implement the `BaseEnv` contract:

- `reset()`
- `get_obs()`
- `step(control_plan)`
- `get_random_control_plan()`

They should preserve the horizon control-plan interface so that controllers can
move between backends without changing policy code.

## Adding a reward

New reward functions should implement `RewardStrategy.compute_reward`. They
receive predicted or realized states, the candidate control plan, forecasts, the
current observation, and an `info` dictionary for component diagnostics.
