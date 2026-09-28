# Replication package

This package contains the processed data, raw static-analysis differences,
analysis code, generated tables, and figures used in the accompanying
SWE-bench Lite patch-quality study.

The package reproduces the quantitative analyses reported in the manuscript
from the included processed inputs. It does not rerun SWE-agent, the SWE-bench
evaluation harness, CodeQL, or CodeScene from source repositories.

## Directory layout

```text
replicate_package/
├── README.md
├── requirements.txt
├── analysis.ipynb                 # full-resolved-set exploratory analysis
├── csv/                           # processed analysis inputs
├── data-0515/                     # per-instance CodeQL/CodeScene diff JSON
├── figures/                       # reference outputs from analysis.ipynb
└── analysis/                      # matched and sensitivity analyses
    ├── 00_direction_share.py
    ├── 01_resolved_sets.py
    ├── 02_matched_static_quality.py
    ├── 03_matched_dynamic_performance.py
    ├── 04_matched_issue_patterns.py
    ├── 05_selection_bias.py
    ├── 06_patch_size_analysis.py
    ├── common.py
    ├── run_all.py
    ├── results/                   # reference tabular outputs
    └── figures/                   # reference analysis figures
```

## Environment

The analyses were last run with Python 3.11.15, NumPy 1.26.4, pandas 2.1.1,
and Matplotlib 3.8.3. To create an isolated environment from the package root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the primary matched and sensitivity analyses

Run the following command from the package root:

```bash
python analysis/run_all.py
```

The command executes the seven numbered scripts in order. It regenerates:

- patch direction shares;
- resolved-set overlap and membership;
- matched CodeQL and CodeScene comparisons;
- matched CPU-time and peak-memory comparisons;
- matched CodeQL-rule and CodeScene-category prevalence;
- resolved-set selection analyses;
- patch-size comparisons and associations.

Outputs are written to `analysis/results/` and
`analysis/figures/`. Randomized procedures use the fixed seed
`20260713`. The run also writes `environment.json` and a SHA-256 manifest of
the regenerated outputs.

## Reproduce the full-resolved-set exploratory analysis

Start Jupyter from the package root and run all cells in `analysis.ipynb`:

```bash
jupyter lab analysis.ipynb
```

The notebook reads the CSV files under `csv/` and writes figures to
`figures/`. Its full-resolved-set comparisons are descriptive sensitivity
analyses; the provider-specific matched analyses in `analysis/` are
the manuscript's primary comparisons.

## Input data

The primary scripts load these processed inputs:

| File | Unit and purpose |
|---|---|
| `csv/codeql_per_instance.csv` | CodeQL before/after and net findings per model-instance |
| `csv/codescene_per_instance.csv` | CodeScene Code Health changes per model-instance |
| `csv/dynamic_perf.csv` | CPU-time and peak-memory aggregates per model-instance |
| `csv/dynamic_perf_runs.csv` | Five repeated resource measurements where available |
| `csv/patch_metrics.csv` | Changed lines, files, hunks, and patch size |
| `csv/rule_catalog.csv` | CodeQL rule metadata and quality categories |

Additional CSV files support the full-resolved-set notebook and provide
descriptive summaries. The `data-0515/` tree preserves the per-instance
`*.diff.json` files used by the matched issue-pattern analysis. Directory
names are retained because `analysis/common.py` maps model names to
those paths. The four `evaluation-*.json` files in the model subdirectories
provide resolved-instance metadata used by `analysis.ipynb`.

## Scope and provenance

The large raw SWE-bench container logs and repository snapshots are excluded
because the included analysis scripts do not read them. Their derived dynamic
measurements are provided in `dynamic_perf_runs.csv` and `dynamic_perf.csv`.
Likewise, full CodeQL SARIF databases are excluded; the package contains the
per-instance comparison JSON and the processed rule catalog required by the
reported analyses.

Model and instance identifiers are retained to support paired comparisons.
The package contains no API keys or local configuration files.
