# Compute Trade-off Results

Profile: `smoke`

| Benchmark | Random final regret | GFlowNet final regret | Random time / trial | GFlowNet time / trial | Slowdown | Regret gain of GFlowNet |
| --- | --- | --- | --- | --- | --- | --- |
| branin | 1.7626 +- 1.1017 | 2.3097 +- 0.9775 | 0.00 min | 0.01 min | 2.56x | -0.5471 |
| hartmann6 | 2.0565 +- 0.6068 | 2.0463 +- 0.5996 | 0.00 min | 0.01 min | 9.15x | 0.0102 |
| ackley10 | 19.7645 +- 0.5366 | 19.7645 +- 0.5366 | 0.00 min | 0.02 min | 10.91x | 0.0000 |

Positive `regret gain of GFlowNet` means the learned policy achieved lower regret than the random baseline.

Recommended report interpretation:
- Report both final regret and wall-clock slowdown.
- Treat Ackley-like positive gains as meaningful only if they justify the extra compute.
- For negative-gain benchmarks, the GFlowNet overhead is not justified under the current reward design.
