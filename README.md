# Sequential Decision Project: v2 Pipeline

This repository now contains a reproducible `pkg`-based pipeline for GFlowNet-guided BO experiments.

The old codebase and old results were moved to `legacy/`.
`project_report/` and PDF files remain in place.
The new pipeline does not overwrite them and writes artifacts separately:

- `artifacts_v2/` — raw run outputs and suite manifests
- `report_assets_v2/` — report-ready figures, tables, and snippets

## Environment

Minimal setup:

```bash
pip install -r requirements.txt
```

Quick sanity check:

```bash
python -m compileall pkg tests scripts
```

## Main Scripts

### 1. Main suite

Smoke-run:

```bash
python scripts/run_v2_suite.py --suite main_v2 --profile smoke
```

Full run:

```bash
python scripts/run_v2_suite.py --suite main_v2 --profile full
```

Outputs:

- one artifact run per benchmark in `artifacts_v2/main_v2/<benchmark>/<timestamp>/`
- a suite manifest in `artifacts_v2/_suites/main_v2/<timestamp>/manifest.json`

### 2. Reward protocol ablation

```bash
python scripts/run_reward_protocol_ablation_v2.py --profile smoke
```

This suite runs:

- `reward_protocol_softplus_scaled_v2`
- `reward_protocol_zscore_v2`
- `reward_protocol_rank_v2`

### 3. Pool ablation

```bash
python scripts/run_v2_suite.py --suite pool_ablation_v2 --profile smoke
```

This suite runs:

- `pool_shared_v2`
- `pool_fresh_v2`

### 4. Finetune ablation

```bash
python scripts/run_v2_suite.py --suite finetune_ablation_v2 --profile smoke
```

This suite runs:

- `finetune_continual_v2`
- `finetune_restart_v2`

## Exporting Report-Ready Artifacts

### Main v2

```bash
python scripts/export_report_assets_v2.py --spec main_v2 --suite-name main_v2
```

Output:

- `report_assets_v2/main_v2/<timestamp>/img/`
- `report_assets_v2/main_v2/<timestamp>/comparison/`
- `report_assets_v2/main_v2/<timestamp>/tables/`
- `report_assets_v2/main_v2/<timestamp>/snippets/`

### Reward protocol ablation

```bash
python scripts/export_reward_protocol_ablation_v2.py
```

Output:

- `report_assets_v2/reward_protocol_ablation_v2/<timestamp>/<spec_name>/...`

Each reward protocol gets its own bundle:

- `img/`
- `comparison/`
- `tables/summary_metrics.csv`
- `tables/summary_metrics.md`
- `snippets/report_snippet.md`

## What To Inspect After A Run

### 1. Main Summary Files

Inside each artifact run:

- `spec.json` — run configuration
- `summary.json` — aggregated results by method
- `trial_metrics.csv` — per-seed metrics

Inside report assets:

- `tables/summary_metrics.md`
- `snippets/report_snippet.md`

### 2. Key Metrics

Look at these first:

- `final_regret_mean`
- `regret_gain_of_gfn`
- `slowdown_factor`
- `floor_reward_fraction_mean`
- `step_reward_std_mean`
- `step_improvement_std_mean`
- `same_mask_repeat_std_mean`
- `proxy_actual_gain_corr_mean`

Interpretation:

- high `regret_gain_of_gfn` — GFN beats the baseline
- high `slowdown_factor` — GFN is more expensive computationally
- high `floor_reward_fraction_mean` — the reward collapses toward a near-constant floor
- high `same_mask_repeat_std_mean` — the reward/evaluation pipeline is noisy
- low or negative `proxy_actual_gain_corr_mean` — the proxy reward is poorly aligned with realized improvement

### 3. Main Figures

Look in `report_assets_v2/.../img/`:

- `*_regret_comparison.png`
- `*_random_diagnostics.png`
- `*_gfn_diagnostics.png`
- `*_reward_hist.png`
- `gfn_reward_hist.png`

What they show:

- `regret_comparison` — final random vs GFN comparison
- `*_diagnostics` — how proxy scores relate to actual BO query outcomes
- `reward_hist` — whether the reward distribution is collapsing

### 4. Comparison With The Old Implementation

Look in `report_assets_v2/.../comparison/`:

- `*_legacy_vs_v2_regret.png`

This is the quickest check for:

whether `v2` is actually more stable than the legacy pipeline.

## Recommended Workflow

For a quick research loop:

1. Run `main_v2` in `smoke`
2. Check `summary.json` and `trial_metrics.csv`
3. Export report assets
4. Inspect `regret_comparison`, `reward_hist`, and `legacy_vs_v2`
5. If the signal looks reasonable, run the needed suite in `full`

For a post-mortem workflow:

1. `main_v2`
2. `pool_ablation_v2`
3. `reward_protocol_ablation_v2`
4. `finetune_ablation_v2`
5. Compare markdown tables and comparison figures

## Current v2 Limitations

Already implemented in the new pipeline:

- shared step context
- reward protocol abstraction
- aligned block-mask training/inference
- script-driven reproducibility
- separate artifact namespaces
- required diagnostics

Not implemented yet:

- synthetic sanity task
- stronger random-top-k baseline
- a dedicated 10-seed full rerun preset
- automatic updates to `project_report/main.tex`

## Useful Paths

- [pkg](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/pkg)
- [scripts](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/scripts)
- [legacy](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/legacy)
- [artifacts_v2](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/artifacts_v2)
- [report_assets_v2](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/report_assets_v2)
