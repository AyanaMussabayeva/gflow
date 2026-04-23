# main_v2 report snippet

branin: random final regret 2.4758 +- 1.8689, GFlowNet final regret 3.0355 +- 1.4537, regret gain of GFN -0.5598, slowdown 32.27x, proxy->actual corr -0.0936 (random) vs 0.0164 (GFN).
hartmann6: random final regret 0.7294 +- 0.5219, GFlowNet final regret 0.8534 +- 0.4978, regret gain of GFN -0.1240, slowdown 47.82x, proxy->actual corr -0.0301 (random) vs 0.1289 (GFN).
ackley10: random final regret 19.5119 +- 0.8592, GFlowNet final regret 18.3047 +- 0.6267, regret gain of GFN 1.2072, slowdown 93.25x, proxy->actual corr -0.2802 (random) vs 0.1606 (GFN).

Diagnostic note: floor reward fraction and same-mask repeat std are included in the CSV/MD tables to quantify reward collapse and reward noise.
