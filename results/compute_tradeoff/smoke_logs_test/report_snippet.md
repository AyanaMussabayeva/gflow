# Compute Trade-off Results

Profile: `smoke`

| Benchmark | Random final regret | GFlowNet final regret | Random time / trial | GFlowNet time / trial | Slowdown | Regret gain of GFlowNet |
| --- | --- | --- | --- | --- | --- | --- |
| branin | 1.9299 +- 1.6452 | 2.9095 +- 1.0385 | 0.01 min | 0.01 min | 1.27x | -0.9797 |

Positive `regret gain of GFlowNet` means the learned policy achieved lower regret than the random baseline.

Recommended report interpretation:
- Report both final regret and wall-clock slowdown.
- Treat Ackley-like positive gains as meaningful only if they justify the extra compute.
- For negative-gain benchmarks, the GFlowNet overhead is not justified under the current reward design.
