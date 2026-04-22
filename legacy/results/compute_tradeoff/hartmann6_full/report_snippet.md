# Compute Trade-off Results

Profile: `full`

| Benchmark | Random final regret | GFlowNet final regret | Random time / trial | GFlowNet time / trial | Slowdown | Regret gain of GFlowNet |
| --- | --- | --- | --- | --- | --- | --- |
| hartmann6 | 0.5571 +- 0.3059 | 0.7027 +- 0.5144 | 0.10 min | 5.18 min | 53.85x | -0.1456 |

Positive `regret gain of GFlowNet` means the learned policy achieved lower regret than the random baseline.

Recommended report interpretation:
- Report both final regret and wall-clock slowdown.
- Treat Ackley-like positive gains as meaningful only if they justify the extra compute.
- For negative-gain benchmarks, the GFlowNet overhead is not justified under the current reward design.
