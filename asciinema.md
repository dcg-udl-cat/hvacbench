# Asciinema Recording Script

This script is for a short terminal demo of `hvacbench`. The recommended first
recording uses only the offline CLI commands, so it does not require a BOPTEST
server or a downloaded TTM checkpoint.

## Before Recording

Run these once before starting the recording so dependency installation and
package build output do not dominate the video:

```bash
uv sync --all-extras --group dev
uv run hvacbench info
uv run hvacbench mock-rollout --steps 2 --history-length 4 --horizon 4
```

Optional warmup if you want to show a real TTM model:

```bash
uv run hvacbench ttm-rollout \
  --model-path gft/ttm4hvac \
  --steps 1 \
  --history-length 8 \
  --horizon 8 \
  --energy-price dynamic
```

## Recording Command

```bash
asciinema rec hvacbench-demo.cast
```

Use a clean terminal with a reasonably large width, for example 100 to 120
columns.

## Main Demo

### 1. Title

Type:

```bash
clear
printf "hvacbench: validating learned HVAC surrogates with receding-horizon environments\n\n"
```

Pause briefly.

### 2. Show What The CLI Exposes

Type:

```bash
uv run hvacbench --help
```

Narration:

> hvacbench provides a small CLI for smoke tests and terminal demos. The offline
> command is useful for quick checks, and the TTM and BOPTEST commands use the
> same receding-horizon control interface.

### 3. Show Project Defaults

Type:

```bash
uv run hvacbench info
```

Narration:

> The default reference building is BOPTEST bestest_air. The CLI also exposes
> the available electricity price profiles: constant, dynamic, and
> highly_dynamic.

### 4. Offline Surrogate Rollout

Type:

```bash
uv run hvacbench mock-rollout --steps 5 --history-length 8 --horizon 8
```

Narration:

> This is the offline smoke test. It uses mock histories and a mock forecasting
> model, then prints the reward, predicted room temperature, HVAC power, and the
> first control action committed at each receding-horizon step.

Pause after the table appears.

### 5. Show TTM Model Selection

Type:

```bash
uv run hvacbench ttm-rollout --help
```

Narration:

> For real learned-surrogate experiments, ttm-rollout lets the user choose a
> TinyTimeMixer-compatible checkpoint or Hugging Face model with --model-path.
> The same command can select the electricity price profile used by the reward.

Do not run the real model command during the short demo unless the model is
already cached and you know it finishes quickly.

## Optional TTM Segment

Use this only if the model is already available locally or cached:

```bash
uv run hvacbench ttm-rollout \
  --model-path gft/ttm4hvac \
  --steps 3 \
  --history-length 8 \
  --horizon 8 \
  --energy-price dynamic
```

Alternative with a local fine-tune:

```bash
uv run hvacbench ttm-rollout \
  --model-path ./models/my-ttm-finetune \
  --steps 3 \
  --history-length 8 \
  --horizon 8 \
  --energy-price highly_dynamic
```

Narration:

> Here the environment is backed by a selected forecasting model. The policy
> still proposes a full horizon, but only the first predicted transition is
> committed before the environment advances.

## Optional BOPTEST Segment

Use this only if BOPTEST is already running with enough workers for the rollout
environment:

```bash
uv run hvacbench boptest-rollout \
  --steps 3 \
  --history-length 8 \
  --horizon 8 \
  --energy-price dynamic \
  --base-url http://127.0.0.1
```

Then show the deployment-style environment:

```bash
uv run hvacbench boptest-evaluate \
  --steps 3 \
  --history-length 8 \
  --horizon 8 \
  --energy-price dynamic \
  --base-url http://127.0.0.1
```

Narration:

> The rollout environment uses BOPTEST to evaluate the proposed horizon. The
> evaluation environment is closer to deployment: it accepts the same control
> plan but applies only the first action.

## Closing

Type:

```bash
printf "\nDone. hvacbench keeps the policy interface fixed while changing the backend used for surrogate validation.\n"
exit
```

After stopping the recording, upload or inspect it:

```bash
asciinema play hvacbench-demo.cast
asciinema upload hvacbench-demo.cast
```
