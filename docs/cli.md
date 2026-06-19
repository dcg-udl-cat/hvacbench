# CLI

`hvacbench` includes a small command-line interface for local smoke tests,
terminal demos, and asciinema recordings.

When working from a source checkout, prefix commands with `uv run`:

```bash
uv run hvacbench info
```

After installation from PyPI or from the local project, the `hvacbench` command
is available directly.

## Offline demo

The `mock-rollout` command runs a deterministic learned-surrogate smoke test
without a trained TTM checkpoint or BOPTEST service:

```bash
uv run hvacbench mock-rollout --steps 5 --history-length 8 --horizon 8
```

This command is the recommended starting point for asciinema because it runs
quickly and does not depend on external services.

## TTM model demo

Use `ttm-rollout` to run the learned-surrogate environment with a real
TinyTimeMixer-compatible checkpoint. Select the checkpoint or Hugging Face model
id with `--model-path`:

```bash
uv run hvacbench ttm-rollout \
  --model-path gft/ttm4hvac \
  --steps 3 \
  --history-length 8 \
  --horizon 8 \
  --energy-price dynamic
```

Use a local fine-tune by pointing `--model-path` to its directory:

```bash
uv run hvacbench ttm-rollout --model-path ./models/my-ttm-finetune
```

This command loads the model and uses the packaged `bestest_air` CSV provider by
default. It requires the selected model to be compatible with the `BaseTTM`
contract used by `TTMEnv`.

## BOPTEST demos

Use `boptest-rollout` when a BOPTEST service is running with enough workers for
two simultaneous `bestest_air` testcase instances:

```bash
uv run hvacbench boptest-rollout \
  --steps 3 \
  --history-length 8 \
  --horizon 8 \
  --base-url http://127.0.0.1
```

Use `boptest-evaluate` for deployment-style evaluation where only the first
proposed action is applied:

```bash
uv run hvacbench boptest-evaluate \
  --steps 3 \
  --history-length 8 \
  --horizon 8 \
  --base-url http://127.0.0.1
```

Both commands accept:

- `--energy-price`: `constant`, `dynamic`, or `highly_dynamic`;
- `--start-day`: BOPTEST initialization day;
- `--heating-setpoint` and `--cooling-setpoint`: constant setpoints used to
  build the demo control plan.

## Energy price modes

The `--energy-price` option controls the price forecast seen by the reward:

- `constant`: a flat electricity price, useful for debugging comfort-focused
  behavior.
- `dynamic`: the default day/night tariff profile.
- `highly_dynamic`: a spot-price-like profile with stronger variation.

For `ttm-rollout`, the option selects the corresponding column from the packaged
CSV electricity-price data. For BOPTEST commands, it selects the BOPTEST
scenario price signal exposed by the `bestest_air` testcase.
