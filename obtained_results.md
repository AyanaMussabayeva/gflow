# Obtained Results

This file summarizes the results obtained by executing the three benchmark notebooks:

- `branin_gfn_bo.ipynb`
- `hartmann6_gfn_bo.ipynb`
- `ackley10_gfn_bo.ipynb`

Each notebook compares two Bayesian optimization policies:

- `random`: block-wise dropout masks sampled uniformly at random.
- `gfn`: a contextual GFlowNet trained to sample masks with probability proportional to proxy improvement potential.

## Final comparison

| Benchmark | Random final regret (mean +- std) | GFlowNet final regret (mean +- std) | Better method |
| --- | --- | --- | --- |
| Branin (2D) | 1.7813 +- 1.1734 | 2.1459 +- 0.7910 | Random |
| Hartmann-6 (6D) | 0.5571 +- 0.3059 | 0.7027 +- 0.5144 | Random |
| Ackley (10D) | 18.1526 +- 1.1661 | 16.7447 +- 1.5421 | GFlowNet |

## Benchmark-by-benchmark observations

### Branin (2D)

On Branin, the learned GFlowNet policy did not beat random mask sampling. The random baseline reached a lower final regret (`1.7813`) than the GFlowNet method (`2.1459`), a gap of `0.3646` regret points in favor of random masks.

One useful detail is that the GFlowNet run had lower variance across seeds (`0.7910` vs `1.1734`). That suggests the learned policy was more consistent, but consistently biased toward weaker proposals than the random baseline.

### Hartmann-6 (6D)

On Hartmann-6, random masks again performed better. The final regret was `0.5571` for random sampling versus `0.7027` for the GFlowNet policy, so the learned sampler underperformed by `0.1456`.

Unlike Branin, the learned policy was also less stable here: the GFlowNet regret standard deviation was `0.5144`, compared with `0.3059` for random masks.

### Ackley (10D)

Ackley-10 was the only benchmark where the learned generative policy improved the final outcome. The GFlowNet method reached a lower final regret (`16.7447`) than random masks (`18.1526`), improving the mean final regret by `1.4079`.

This is the main positive signal in the current experiments. It suggests that learned mask generation may become more useful in higher-dimensional settings where uniform random mask sampling is less effective.

## Overall interpretation

The current results are mixed rather than uniformly positive:

- On the lower-dimensional benchmarks (Branin and Hartmann-6), random mask sampling remained stronger.
- On the higher-dimensional Ackley-10 task, the learned GFlowNet policy outperformed the random baseline.

This pattern is consistent with a reasonable project hypothesis: the additional cost of learning a mask policy may not be justified on simpler low-dimensional landscapes, but it can start to pay off once the search space becomes harder and more structured.

At the same time, these runs do not yet show a clear across-the-board win for GFlowNet-guided BO. A fair reading is:

- the method is promising in higher dimensions,
- it still needs better reward design or stronger surrogate calibration on easier benchmarks,
- and the computational overhead is substantial.

## Runtime note

The notebooks executed successfully in `general_env`, but the computational cost was significant on this machine:

- Branin: about 11 minutes
- Hartmann-6: about 20 minutes
- Ackley-10: about 48 minutes

This supports the assignment's computational trade-off question directly: the GFlowNet-based procedure is feasible, but the overhead grows quickly with benchmark difficulty.
