# Prompt for Prism OpenAI Agent: Draft the SoftwareX Manuscript

You are writing a SoftwareX "Original software publication" manuscript for the
Python library `hvacbench`.

Use the repository as the source of truth. Inspect the code, tests, examples,
documentation, and `PUBLICATION_TODO.md` before writing. Do not invent results,
DOIs, authors, affiliations, funding, or release identifiers. Use TODO markers
where final publication metadata is missing.

## Software context

`hvacbench` is a Python library for building HVAC control research. It provides
a shared receding-horizon environment API for comparing environment backends:

- `TTMEnv`: learned digital-twin backend using TinyTimeMixer-compatible models.
- `BoptestRolloutEnv`: BOPTEST-backed horizon rollout backend with a committed
  simulator and a shadow rollout simulator.
- `BoptestEvaluationEnv`: single-simulator BOPTEST backend for realized policy
  evaluation.

The software is meant to make controller experiments more comparable by
decoupling controller logic from the backend used for rollout and evaluation.
The current implementation includes a `bestest_air` BOPTEST mapping, packaged
CSV data providers, a simple reward, safety filtering, mock backends for tests,
and examples.

## Manuscript positioning

Frame the paper around these points:

1. Building control research often mixes controller code, simulator backends,
   forecasts, and rewards in one experiment script.
2. `hvacbench` provides a small, reusable environment contract for
   receding-horizon HVAC control.
3. The same policy-facing API can be used with a learned TinyTimeMixer digital
   twin and BOPTEST-backed backends.
4. This enables clearer comparisons between surrogate training, horizon rollout
   scoring, and realized policy evaluation.
5. Although the current BOPTEST mapping targets `bestest_air`, the software is
   structured around extensible providers, models, rewards, and backend
   clients.

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
   - Explain the need for comparable HVAC control environment backends.
   - Explain why learned digital twins and physics-based BOPTEST evaluation need
     a common policy-facing interface.

3. **Software description**
   - Describe architecture: envs, providers, models, rewards, safety, BOPTEST
     clients, schemas.
   - Explain the receding-horizon step contract.
   - Explain the difference between rollout and evaluation BOPTEST backends.

4. **Illustrative examples**
   - Use examples from `examples/` and tests.
   - Do not claim performance results unless they are present in the repository.

5. **Impact**
   - Focus on reproducible controller comparison, digital-twin evaluation, and
     backend interchangeability.
   - Avoid overselling generality beyond the current implementation.

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
- Python package dependencies when relevant.

Do not fabricate citations. If a reference cannot be verified, leave a TODO.

## Output requirements

Produce a complete first manuscript draft, but mark uncertain publication
metadata with TODO. Keep language precise and suitable for a multidisciplinary
research-software audience.
