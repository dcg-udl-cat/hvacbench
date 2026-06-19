# Publication TODO

Tasks to complete before SoftwareX submission and public release.

## Repository

- [ ] Set `origin` to `git@github.com:dcg-udl-cat/hvacbench.git`.
- [ ] Push the current `main` branch to the new repository.
- [ ] Confirm GitHub Actions pass on the new repository.
- [ ] Enable GitHub Pages with the Actions deployment workflow.
- [ ] Confirm `https://dcg-udl-cat.github.io/hvacbench/` renders correctly.
- [ ] Review README badges after the first CI and Pages runs.

## Release and citation

- [ ] Decide final authors, affiliations, ORCIDs, and corresponding author.
- [ ] Add `CITATION.cff` with final authors and release metadata.
- [ ] Create the first GitHub release.
- [ ] Archive the GitHub release on Zenodo and obtain a DOI.
- [ ] Add DOI badge and citation instructions to README and docs.
- [ ] Review the direct Git dependency on `granite-tsfm` before publishing to
      PyPI, and replace it with an index-resolvable dependency or optional
      installation path if needed.
- [ ] Publish the package to PyPI.
- [ ] Add a PyPI badge after the package is public.

## SoftwareX manuscript

- [ ] Download the current SoftwareX original software publication template.
- [ ] Draft the manuscript from `SOFTWAREX_MANUSCRIPT_PROMPT.md`.
- [ ] Keep the manuscript within the 3000-word SoftwareX limit.
- [ ] Prepare 3 to 5 highlights, each no more than 85 characters.
- [ ] Prepare a concise abstract no longer than 250 words.
- [ ] Fill in keywords, author affiliations, funding statement, competing
      interests, data availability, and generative-AI declaration.
- [ ] Complete the SoftwareX software metadata table with repository URL,
      license, version, operating system, dependencies, and archival DOI.

## Technical validation

- [ ] Run `uv lock --check`.
- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check src tests examples`.
- [ ] Run `uv run mkdocs build --strict`.
- [ ] Run `uv build` and inspect the generated source distribution and wheel.
- [ ] Run `examples/run_mock_env.py`.
- [ ] Run a live BOPTEST smoke test for `BoptestRolloutEnv`.
- [ ] Run a live BOPTEST smoke test for `BoptestEvaluationEnv`.
