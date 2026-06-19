# Prompt for Prism OpenAI Agent: Draft the SoftwareX Manuscript

You are writing a SoftwareX "Original software publication" manuscript for the
Python library `hvacbench` which resides in `https://github.com/dcg-udl-cat/hvacbench`.

Use the repository as the source of truth. Inspect the code, tests, examples,
documentation, and `PUBLICATION_TODO.md` before writing. Do not invent results,
DOIs, authors, affiliations, funding, or release identifiers. Use TODO markers
where final publication metadata is missing.

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
providers derived from BOPTEST operation, a simple reward, safety filtering,
mock backends for tests, and examples.

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

## Required SoftwareX constraints to respect

Follow the current SoftwareX guide for authors:

- The paper is a short descriptive paper with a 3000-word limit.
- Use the official SoftwareX original software publication template.
- Include title page information, abstract, keywords, and highlights.
- Keep the abstract under 250 words.
- Prepare 3 to 5 highlights, each no more than 85 characters.
- Include funding, competing interests, data availability, and generative-AI
  declarations as required by Elsevier.
- Include the SoftwareX software metadata table.

## Sections to draft

Draft these sections in SoftwareX style:

1. **Metadata table**
   - Current code version: TODO
   - Permanent link to code/repository: https://github.com/dcg-udl-cat/hvacbench
   - Permanent link to reproducible capsule: TODO if any
   - Legal code license: MIT
   - Code versioning system: git
   - Software code languages, tools, and services: Python, uv, BOPTEST, TinyTimeMixer
   - Compilation requirements, operating environments, and dependencies: derive
     from `pyproject.toml`
   - If available, link to developer documentation/manual: GitHub Pages URL TODO
   - Support email: TODO

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

4. **Illustrative examples**
   - Use examples from `examples/` and tests.
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

Produce a complete first manuscript draft, but mark uncertain publication
metadata with TODO. Keep language precise and suitable for a multidisciplinary
research-software audience.

## Author's view on the project

Here is how the author describes the project:

Surrogate building models where learning agents can train their control
policies are expensive and complex to build if physics compoments are
modeled. Instead, surrogates learned from operation data of buildings are
easier to get, but not as realiable. Even more, if these are based on deep
learning models which are black box in nature.

The purpose of this work is to facilitate validation of such surrogate
models learned from operational data, making it easy to train a policy on
a gym-like environment that is backed by these models, and then also
evaluate that policy on environments backed by physics based simulator
BOPTEST treating it as if it was a real building.

In this project we already implement an environment backed by any
forecasting model derived from base model TTM (TinyTimeMixer from IBM,
which is a Foundation Time Series Forecasting model that can forecast
zero-shot or be easily fine-tunned) as long as it has been fine-tuned to
forecast rollouts on a horizon. The model is expected to take a history of
observations plus a future control plan along a given horizon size and
output the forecasted states of the building if those actions where to be
applied, then only the first predicted states is kept, creating a
receding-horizon scenario. This TTMEnv, by default uses data generated
from the operation of BOPTEST's Bestest Air test case. We also implement a
BOPTEST backed env which expects the same control plans into the horizon
and where we have made it possible through two concurrent BOPTEST clients
to generate rollouts and know the resulting states of applying that
control plan, then we only commit the first proposed action and create the
same receding-horizon scenario. This BOPTEST backed backend also uses
Bestst Air test case by default.

Envs receive a reward object which will define how the reward is computed
given the proposed control plan and future states that control plan would
give, the reward object also receives energy price and weather
observations.

There is also a second BOPTEST backed environment where the control plan
is not rolled out and only the first action is applied, as if we where
actually deploying the control policy in a real building where we cannot
rollout the entire control plan but just commit the first proposed action.
This environemnt also uses Bestest Air by default.

As provided, the project easily enables a control policy to be trained on
a data driven learned surrogate which may not trust yet and then see how
it behaves on a physics based backed env that we do trust and treaet as
the real building. All of this is done in a reciding-horizon scenario and
with the purpose of enabling easy validation of data driven learned
surrogates and rececing-horizon control policies. The second BOPTEST
backed env extends the set of experiments one can do on this setup. What
the user is expected to provide is a finetune of a TTM model. For first
steps we refer to https://huggingface.co/gft/ttm4hvac where there is
already a collection of finetunes that will work with this project.

The project has been designed with the idea that more envs can be easily
added, expecting future work to include more simulator backed envs that
could use Sinergym for example. Also, one can easily do its own fine tune
of a TTM and use it within this project to easily and quickly validate it
as valid surrogate model. Other kind of forecasting models other than TTM
could easily be used and TTM is just the one we alredy provide.
