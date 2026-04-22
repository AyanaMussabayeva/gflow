# Compute Trade-off Results

Profile: `full`

| Benchmark | Random final regret | GFlowNet final regret | Random time / trial | GFlowNet time / trial | Slowdown | Regret gain of GFlowNet |
| --- | --- | --- | --- | --- | --- | --- |
| branin | 1.7813 +- 1.1734 | 2.1459 +- 0.7910 | 0.06 min | 2.13 min | 34.11x | -0.3646 |

Positive `regret gain of GFlowNet` means the learned policy achieved lower regret than the random baseline.

Recommended report interpretation:
- Report both final regret and wall-clock slowdown.
- Treat Ackley-like positive gains as meaningful only if they justify the extra compute.
- For negative-gain benchmarks, the GFlowNet overhead is not justified under the current reward design.
