# Compute Trade-off Results

Profile: `full`

| Benchmark | Random final regret | GFlowNet final regret | Random time / trial | GFlowNet time / trial | Slowdown | Regret gain of GFlowNet |
| --- | --- | --- | --- | --- | --- | --- |
| ackley10 | 18.1526 +- 1.1661 | 16.7447 +- 1.5421 | 0.13 min | 8.49 min | 67.70x | 1.4079 |

Positive `regret gain of GFlowNet` means the learned policy achieved lower regret than the random baseline.

Recommended report interpretation:
- Report both final regret and wall-clock slowdown.
- Treat Ackley-like positive gains as meaningful only if they justify the extra compute.
- For negative-gain benchmarks, the GFlowNet overhead is not justified under the current reward design.
