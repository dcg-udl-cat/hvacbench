# Contributing

Thank you for considering a contribution to `hvacbench`.

## Local setup

```bash
uv sync --all-extras --group dev
```

## Checks

Run these before opening a pull request:

```bash
uv lock --check
uv run ruff check src tests examples
uv run pytest
uv run mkdocs build --strict
```

For packaging changes, also run:

```bash
uv build
```

## Development guidelines

- Keep environment behavior backend-agnostic whenever possible.
- Add or update tests when changing environment semantics, providers, rewards,
  or BOPTEST mappings.
- Keep examples small enough to run as smoke tests.
- Do not include experiment-specific training runs, generated results, or large
  model artifacts in the library repository.
