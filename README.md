# PRISM portfolio concentration simulation

DOI: 10.5281/zenodo.22178964

[![Tests](https://github.com/MRDOANE/prism-portfolio-concentration/actions/workflows/tests.yml/badge.svg)](https://github.com/MRDOANE/prism-portfolio-concentration/actions/workflows/tests.yml)

This repository contains the simulation code, prespecified configurations,
reference results, and numerical validation records for:

> **Beyond Independent Shots on Goal: Risk Capacity, Synergy, and
> Concentration in Pharmaceutical R&D Portfolios**

Michael R. Doane, D.Eng.  
Independent Researcher, Cary, North Carolina, USA  
[ORCID 0009-0003-0521-8981](https://orcid.org/0009-0003-0521-8981)

## What the model evaluates

The analysis compares matched pharmaceutical R&D portfolios arranged as:

- independent assets;
- a pipeline in a pill (PIAP), with one asset developed across indications; or
- multiple shots on goal (MSOG), with assets linked by indication, target, or
  pathway.

Technical dependence is separated from R&D cost synergy, development timing,
and commercial cannibalization. Outputs include expected net present value
(eNPV), expected profitability index (ePI), fifth-percentile NPV, CVaR5,
financing-capacity breach, effective breadth, rolling three-year launch
service, and decision regret.

## Headline reference results

- Six PIAP programs provided 3.19 variance-equivalent independent shots at the
  empirical dependence anchor.
- The median PIAP compensating R&D-synergy threshold was 23% across 1,000
  matched heterogeneous opportunity sets. The 10th to 90th percentile range
  was 15% to 29%.
- Seventeen percent of sampled PIAP portfolios had no feasible solution through
  30% follower R&D saving under the high-capacity balanced policy.
- Staging saved a median of $114 million in development cost while reducing
  eNPV by $428 million and increasing financing-capacity breach by 12.0
  percentage points.
- An 80% service target of at least two launches per rolling three-year period
  required 19 independent starts or 25 PIAP starts under the benchmark.
- Selecting concentration under additive revenue and then experiencing 25%
  cannibalization produced median eNPV regret of $59 million at 10% cost
  saving.

All monetary values are constant 2026 US dollars in millions. These are
decision-model outputs under stated assumptions, not forecasts for a named
company or product.

## Repository structure

```text
config/          Locked benchmark and experiment plans
prism_sim/       Simulation and analytical library
scripts/         Experiment, analysis, and reproduction entry points
tests/           Unit and numerical tests
results_e4_e7/   Synergy, staging, launch-service, and regret results
results_e8/      Matched-heterogeneity results and boundary confirmation
docs/            Release metadata and journal-formatted manuscript files
qc/              Manuscript quality-control records retained for provenance
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install the locked dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verification and reproduction

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Reproduce the E4-E7 decision-frontier experiments:

```bash
python scripts/reproduce_e4_e7.py --output outputs_e4_e7_reproduced
```

Reproduce the E8 matched-heterogeneity analysis:

```bash
python scripts/reproduce_e8.py --output outputs_e8_reproduced
```

The E8 analysis evaluates all 4^6 realized program states using Gauss-Hermite
integration for each of 1,000 Latin hypercube opportunity sets. Eight
near-boundary cells are independently confirmed with 250,000 Monte Carlo paths
each. The complete reference outputs are included so the reported results can
be inspected without rerunning the full experiment.

The `docs/manuscript` directory contains the TIRS-formatted research article
and supplementary material associated with release `v1.0.0`. Submission-only
materials such as the cover letter are intentionally excluded.

## Interpretation limits

The flat benchmark is designed to isolate portfolio topology and mechanism
effects. The same-asset latent correlation of 0.415 is a scenario anchor, and
commercial cannibalization is represented by a bounded stress function. The
enterprise layer requires sponsor-specific calibration before use in an
organizational decision process.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Zenodo will
register a version-specific DOI when the first GitHub release is archived. The
DOI should be used when citing the software and reference results.

## License

The source code and repository materials are released under the [MIT
License](LICENSE).

## AI-assistance disclosure

Generative AI tools from OpenAI were used under the author's direction to
assist with code generation, quality control, literature organization, figure
preparation, and language editing. The author controlled the analytical
specifications, reviewed the code, verified the outputs, interpreted the
results, and approved the final repository contents. No generative AI was used
to create simulated data or images.
