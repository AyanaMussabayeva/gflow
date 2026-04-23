# main_v2 report snippet

branin: random final regret 0.5227 +- 0.4245, GFlowNet final regret 0.5227 +- 0.4245, regret gain of GFN 0.0000, slowdown 0.96x, proxy->actual corr 0.3460 (random) vs -0.0158 (GFN).
hartmann6: random final regret 2.4337 +- 0.3893, GFlowNet final regret 1.6010 +- 0.4434, regret gain of GFN 0.8327, slowdown 7.84x, proxy->actual corr -0.3787 (random) vs 0.4203 (GFN).
ackley10: random final regret 19.3961 +- 0.7346, GFlowNet final regret 19.3961 +- 0.7346, regret gain of GFN 0.0000, slowdown 9.65x, proxy->actual corr 0.3833 (random) vs 0.0137 (GFN).

Diagnostic note: floor reward fraction and same-mask repeat std are included in the CSV/MD tables to quantify reward collapse and reward noise.
