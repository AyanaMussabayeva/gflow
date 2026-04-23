# main_v2 report snippet

branin: random final regret 1.9766 +- 1.8077, GFlowNet final regret 1.9087 +- 1.8086, regret gain of GFN 0.0679, slowdown 1.64x, proxy->actual corr -0.3274 (random) vs -0.1266 (GFN).
hartmann6: random final regret 2.3922 +- 0.4492, GFlowNet final regret 2.2768 +- 0.3654, regret gain of GFN 0.1154, slowdown 6.96x, proxy->actual corr 0.1750 (random) vs 0.1213 (GFN).
ackley10: random final regret 20.4990 +- 0.2548, GFlowNet final regret 20.4990 +- 0.2548, regret gain of GFN 0.0000, slowdown 8.63x, proxy->actual corr -0.4938 (random) vs -0.1705 (GFN).

Diagnostic note: floor reward fraction and same-mask repeat std are included in the CSV/MD tables to quantify reward collapse and reward noise.
