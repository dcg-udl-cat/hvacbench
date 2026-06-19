# SoftwareX readiness

`hvacbench` is being prepared for submission to SoftwareX as reusable research
software for validating learned HVAC surrogate models in receding-horizon
control workflows.

## Repository requirements

The SoftwareX guide states that submissions include a short descriptive paper
and an open-source software distribution with support material. It also states
that accepted software publications are hosted on GitHub for archival purposes.
This repository is therefore prepared with:

- source code under `src/hvacbench`;
- `README.md` with installation, usage, and testing instructions;
- `LICENSE.txt`;
- automated test and documentation workflows;
- examples and unit tests;
- a publication TODO checklist.

## Contribution to emphasize

The manuscript should frame `hvacbench` as a validation harness for
operation-data-driven building surrogates. The central contribution is not only
that multiple backends exist, but that the same receding-horizon policy can be
trained on a learned forecasting model and then evaluated against BOPTEST
environments treated as trusted physics-based references.

## Manuscript requirements to complete later

Before submission, the authors still need to prepare the SoftwareX manuscript
using the official template. The current guide specifies:

- a 3000-word limit for the descriptive paper;
- SoftwareX-specific templates;
- title page, abstract, keywords, and highlights;
- funding and competing-interest declarations;
- declaration of generative AI use when applicable.

The repository includes `SOFTWAREX_MANUSCRIPT_PROMPT.md` to help a writing agent
draft the manuscript from the validated software state.
