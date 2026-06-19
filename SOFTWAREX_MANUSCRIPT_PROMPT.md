# Prompt for Prism OpenAI Agent: Draft the SoftwareX Manuscript

You are writing a SoftwareX "Original software publication" manuscript for the
Python library `hvacbench` which resides in `https://github.com/dcg-udl-cat/hvacbench`.

Use the repository as the source of truth. Inspect the code, tests, examples,
documentation, `TODO.md`, and the local SoftwareX LaTeX template
`softwarex-osp-template.tex` before writing. Do not invent results, DOIs,
authors, affiliations, funding, support emails, or release identifiers. Use TODO
markers where final publication metadata is missing.

Use `softwarex-osp-template.tex` as the structural authority for the manuscript.
Create a new manuscript file, for example `softwarex-manuscript.tex`, rather
than overwriting the template. Preserve the `elsarticle` structure and mandatory
SoftwareX sections. Delete the template's instructional "Before you begin"
section from the actual manuscript.

## Software context

`hvacbench` is a Python library for validating learned building surrogate
models in receding-horizon HVAC control research. It provides a shared
environment API so a policy can be trained on a data-driven forecasting model
and then evaluated on BOPTEST-backed environments treated as trusted
physics-based references:

- `TTMEnv`: learned-surrogate backend using TinyTimeMixer-compatible models.
- `BoptestRolloutEnv`: BOPTEST-backed horizon rollout backend with a committed
  simulator and a shadow rollout simulator.
- `BoptestEvaluationEnv`: single-simulator BOPTEST backend for realized policy
  evaluation.

The software is meant to make learned surrogate models easier to validate for
control-oriented use. Surrogates learned from building operation data are easier
to obtain than high-fidelity physics models, but they are less reliable and are
often black-box deep learning models. `hvacbench` enables a user to train a
policy on such a surrogate, then evaluate whether the policy behaves well on
BOPTEST-backed environments that act as the reference building. The current
implementation includes a `bestest_air` BOPTEST mapping, packaged CSV data
providers derived from BOPTEST operation, configurable electricity-price
scenarios, a simple reward, safety filtering, mock backends for tests, examples,
and a CLI for smoke tests and terminal demos.

## Manuscript positioning

Frame the paper around these points:

1. Physics-based building surrogate models can be expensive and complex to
   construct when detailed thermal and HVAC components are modeled.
2. Surrogates learned from operational data are easier to obtain, but their
   black-box nature makes them difficult to trust for control-policy training.
3. `hvacbench` provides a receding-horizon validation workflow: train a policy
   on a learned surrogate and evaluate it on BOPTEST-backed environments.
4. `TTMEnv` supports TinyTimeMixer-compatible forecasting models that receive
   histories and future control plans and forecast state rollouts.
5. `BoptestRolloutEnv` produces comparable horizon rollouts through two
   synchronized BOPTEST clients, while `BoptestEvaluationEnv` provides the
   deployment-like case where only the first action is applied.
6. The current implementation targets BOPTEST `bestest_air`, but the structure
   supports additional simulator-backed environments, BOPTEST testcases, reward
   functions, and forecasting models.
7. The package includes a command-line interface that can run offline mock
   rollouts, real TTM-backed rollouts with user-selected model paths, and live
   BOPTEST smoke tests, supporting asciinema demonstrations and quick
   reproducibility checks.

## Required SoftwareX constraints to respect

Follow the local SoftwareX original software publication template:

- Use the `elsarticle` LaTeX structure from `softwarex-osp-template.tex`.
- The main text has a 4000-word limit and a maximum of 6 pages excluding
  metadata, tables, figures, and references.
- Replace the template's instructional `enumerate` block with the five
  mandatory main-text sections from the template:
  Motivation and significance, Software description, Illustrative examples,
  Impact, and Conclusions.
- Include the mandatory C1-C8 code metadata table exactly as a SoftwareX
  metadata table, keeping the left-column labels from the template.
- Include the "Current executable software version" section at the end of the
  manuscript.
- Include title page information, abstract, and keywords in the LaTeX front
  matter.
- Keep the abstract near the template's requested length of about 100 words.
- If Elsevier's submission system separately asks for highlights, prepare 3 to
  5 highlights no more than 85 characters each outside the LaTeX body unless the
  template is updated to include them.
- Include funding, competing interests, data availability, and generative-AI
  declarations as required by Elsevier, using TODO markers when final
  statements are not known.

## Sections to draft

Draft these sections in SoftwareX style and map them to the LaTeX template:

1. **Metadata table**
   - Current code version: TODO
   - Permanent link to code/repository: https://github.com/dcg-udl-cat/hvacbench
   - Legal code license: MIT
   - Code versioning system: git
   - Software code languages, tools, and services: Python, uv, BOPTEST,
     TinyTimeMixer, command-line interface
   - Compilation requirements, operating environments, and dependencies: derive
     from `pyproject.toml`
   - If available, link to developer documentation/manual:
     https://dcg-udl-cat.github.io/hvacbench/
   - Support email: TODO
   - Keep the C1-C8 rows from `softwarex-osp-template.tex`; do not add extra
     rows unless the template is updated.

2. **Motivation and significance**
   - Explain why learned building surrogates from operational data are useful
     but need validation before they are trusted for control-policy training.
   - Explain why a common policy-facing interface is needed to compare training
     on a learned surrogate with evaluation on BOPTEST.
   - Position BOPTEST as a physics-based reference environment, not as the
     object being validated.

3. **Software description**
   - Describe architecture: envs, providers, models, rewards, safety, BOPTEST
     clients, schemas.
   - Explain the receding-horizon step contract.
   - Explain `TTMEnv`: histories plus future control plan go into the model;
     the model forecasts state rollouts; only the first transition is committed.
   - Explain `BoptestRolloutEnv`: two BOPTEST clients allow comparable horizon
     rollouts before committing only the first action.
   - Explain `BoptestEvaluationEnv`: only the first action is applied, matching
     deployment in a real building where future rollouts cannot be queried.
   - Explain that reward objects receive future states, control plans, weather,
     energy prices, and current observations.
   - Explain `EnergyPriceType`: `constant`, `dynamic`, and `highly_dynamic`
     profiles are exposed to observations and rewards; the enum selects packaged
     CSV price columns for `TTMEnv` and BOPTEST scenario price signals for
     BOPTEST-backed environments.
   - Explain that the CLI exposes `info`, `mock-rollout`, `ttm-rollout`,
     `boptest-rollout`, and `boptest-evaluate` commands for demonstrations and
     smoke tests. `ttm-rollout --model-path ...` lets users choose the
     TinyTimeMixer-compatible checkpoint or Hugging Face model id.

4. **Illustrative examples**
   - Use examples from `examples/`, tests, and the CLI documentation.
   - Include a short terminal-style example based on
     `hvacbench mock-rollout --steps 5 --history-length 8 --horizon 8`; this is
     suitable for an asciinema recording because it has no external service
     dependency.
   - Include a second CLI example showing model selection:
     `hvacbench ttm-rollout --model-path gft/ttm4hvac --energy-price dynamic`.
   - Do not claim performance results unless they are present in the repository.

5. **Impact**
   - Focus on control-oriented validation of learned building surrogates,
     receding-horizon policy evaluation, and policy transfer from learned models
     to physics-based reference environments.
   - Discuss extensibility as future-facing: more BOPTEST testcases,
     Sinergym-like simulator-backed environments, and other forecasting models
     can be added.
   - Avoid overselling generality beyond the current `bestest_air` and
     TinyTimeMixer-backed implementation.

6. **Conclusions**
   - Summarize what `hvacbench` enables now and what extension paths it supports.

7. **Declarations**
   - Funding: TODO
   - Competing interests: TODO
   - Data availability: packaged CSV data and repository URL; clarify any
     external BOPTEST/model requirements.
   - Generative AI declaration: TODO based on actual writing workflow.
   - The local template does not provide a dedicated declarations block. If
     these statements are required for submission, place them near
     Acknowledgements or immediately before References, and keep TODO markers
     for unknown final statements.

8. **Current executable software version**
   - Fill the template's final "Current executable software version" section.
   - Include executable/package name: `hvacbench`.
   - Include version: TODO until final release tag is created.
   - Include source code repository: https://github.com/dcg-udl-cat/hvacbench
   - Include license: MIT.
   - Include installation/execution notes from README and CLI docs.

## References and sources to verify

Use real, verified references for:

- SoftwareX guide/template.
- BOPTEST.
- BESTEST Air testcase, if cited separately.
- TinyTimeMixer / Granite TSFM.
- `gft/ttm4hvac` and related TTM4HVAC model/dataset pages.
- Python package dependencies when relevant.
- Other frameworks we may need to compare this one to.

Do not fabricate citations. If a reference cannot be verified, leave a TODO.

## Output requirements

Produce a complete first LaTeX manuscript draft compatible with
`softwarex-osp-template.tex`, but mark uncertain publication metadata with TODO.
Keep language precise and suitable for a multidisciplinary research-software
audience. Do not include the template's instructional boilerplate in the final
manuscript.

## Author's project framing

Use this framing when drafting the manuscript:

Physics-based building models for control-policy training are expensive and
complex to construct when detailed building, thermal, and HVAC components must
be modeled. Data-driven surrogates learned from building operation data are
easier to obtain, but they are less reliable and harder to trust, especially
when the surrogate is a black-box deep learning forecaster.

The purpose of `hvacbench` is to facilitate validation of those learned
surrogates. A user can train a policy in a Gym-like environment backed by a
forecasting model, then evaluate the same policy in BOPTEST-backed environments
treated as a trusted physics-based proxy for the real building.

The implemented learned-surrogate path is `TTMEnv`. It supports forecasting
models compatible with the `BaseTTM` interface, including TinyTimeMixer
fine-tunes. The model receives histories plus a future control plan and outputs
forecasted building states over the horizon. `TTMEnv` scores the horizon but
commits only the first predicted transition, creating a receding-horizon
scenario. By default, the packaged data provider uses data generated from
BOPTEST `bestest_air` operation. The CLI command
`hvacbench ttm-rollout --model-path ...` should be described as the practical
way to select a TTM fine-tune for a terminal demonstration.

The BOPTEST rollout path is `BoptestRolloutEnv`. It expects the same
full-horizon control plans and uses two concurrent BOPTEST clients: one for the
committed simulator and one for horizon rollouts. After scoring the proposed
plan, only the first action is committed, matching the same receding-horizon
semantics as `TTMEnv`.

The deployment-style path is `BoptestEvaluationEnv`. It accepts the same
control plan shape but does not roll out the horizon. It applies only the first
action and computes reward from the realized one-step transition, matching the
case where a controller is deployed on a building and cannot query future
rollouts.

Rewards are supplied by the user. A reward object receives the proposed control
plan, forecasted or realized future states, weather observations, energy-price
forecasts, the current observation, and diagnostic information.

Electricity prices are represented with `EnergyPriceType`. The available modes
are `constant`, `dynamic`, and `highly_dynamic`. These modes let examples and
experiments change the reward signal without changing the environment contract.

For first steps, refer users to https://huggingface.co/gft/ttm4hvac. Users may
also fine-tune their own TTM model or adapt another forecasting model to the
`BaseTTM` interface. Future work can add more BOPTEST testcases and additional
simulator-backed environments, including Sinergym-like backends.
