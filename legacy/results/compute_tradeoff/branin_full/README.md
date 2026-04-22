# Compute Trade-off Experiment

This directory stores the outputs of the compute trade-off experiment comparing random mask sampling and the contextual GFlowNet policy.

Configuration:
- profile: `full`
- benchmarks: `branin`
- seeds: `0, 1, 2, 3, 4`

Files:
- `raw_results.json`: raw trial outputs with regrets and timing arrays
- `trial_metrics.csv`: one row per seed/method/benchmark
- `summary_metrics.csv`: aggregated metrics per method and benchmark
- `tradeoff_summary.csv`: direct random-vs-GFlowNet trade-off metrics
- `compute_tradeoff_overview.png`: runtime and regret overview figure
- `compute_tradeoff_breakdown.png`: per-iteration runtime breakdown figure
- `report_snippet.md`: ready-to-paste report summary
