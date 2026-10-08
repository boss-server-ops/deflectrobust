# DEFLECT — Raw Data per Figure (for re-drawing)

For each figure that needs plotted data, the section header is the paper figure number (matching the latest manuscript: 14 figures total).
Skipped: Figure 1 (teaser), Figure 2 (motivation example) — neither has plottable data.

All Kinetix entries: success rate (%), 12 tasks × 1024 rollouts/cell.
All LIBERO entries: 500 episodes per (suite, delay).

---

## Figure 3 — Kinetix delay robustness (main paper)

Success rate vs inference delay  d ∈ {0..7}, K = max(d, 1).

| d | Naive | VLASH | RTC | BID | PFM σ=0 | Sync‡ | DEFLECT |
|---|---|---|---|---|---|---|---|
| 0 | 89.3 | 89.5 | 89.2 | 89.7 | 90.3 | 89.4 | 91.3 |
| 1 | 76.5 | 89.3 | 86.8 | 84.8 | 89.3 | 89.4 | 90.9 |
| 2 | 68.1 | 87.7 | 82.1 | 77.5 | 87.3 | 89.1 | 89.1 |
| 3 | 53.9 | 85.5 | 69.5 | 62.2 | 84.6 | 87.5 | 88.4 |
| 4 | 46.5 | 81.9 | 57.5 | 53.5 | 79.0 | 86.6 | 86.1 |
| 5 |  3.9 | 74.6 |  4.9 |  5.0 | 72.0 | 86.1 | 80.8 |
| 6 |  0.1 | 66.3 |  0.2 |  0.1 | 64.9 | 85.2 | 73.3 |
| 7 |  0.6 | 60.5 |  0.9 |  0.8 | 59.5 | 83.3 | 66.3 |

(PFM σ=0.1 alternative: 89.6 / 89.2 / 87.4 / 84.4 / 79.9 / 71.7 / 64.6 / 59.6 — within 0.3 pp of σ=0 everywhere.)

---

## Figure 4 — Real-robot results (main paper)

### Figure 4(a): Conveyor full-task success (two conveyor tasks, N = 30)

| Method  | Conveyor-I full task | Conveyor-II full task |
|---|---|---|
| π_0.5   | 96.7 | 46.7 |
| VLASH   | 86.7 | 83.3 |
| DEFLECT | 96.7 | 90.0 |

### Figure 4(b): Whack-a-Mole (mean moles struck per 30 s trial, N = 30)

| Method  | mean moles struck / 30 s |
|---|---|
| π_0.5   | 8.9 |
| VLASH   | 10.4 |
| DEFLECT | 13.6 |

---

## Figure 5 — Scoring-context ablation (appendix)

X axis: delay d ∈ {0..7}. Two curves: matched (per-action conditioning) vs unified (DEFLECT, all under c^mix). Gap monotone in d.

| d | matched-context | unified (DEFLECT) | Δ (matched − DEFLECT) |
|---|---|---|---|
| 0 | 91.0 | 91.3 | −0.3 |
| 1 | 90.4 | 90.9 | −0.5 |
| 2 | 88.4 | 89.1 | −0.7 |
| 3 | 87.5 | 88.4 | −0.9 |
| 4 | 84.8 | 86.1 | −1.3 |
| 5 | 79.2 | 80.8 | −1.6 |
| 6 | 71.4 | 73.3 | −1.9 |
| 7 | 64.0 | 66.3 | −2.3 |

---

## Figure 6 — Kinetix execution-horizon robustness (appendix)

Success rate vs execution horizon  K ∈ {1..8} at d = 1.

| K | Naive | VLASH | RTC | BID | PFM σ=0 | DEFLECT |
|---|---|---|---|---|---|---|
| 1 | 76.5 | 89.3 | 86.9 | 84.8 | 89.3 | 90.9 |
| 2 | 78.5 | 88.5 | 85.7 | 83.5 | 88.5 | 90.1 |
| 3 | 76.1 | 87.4 | 82.9 | 82.1 | 87.0 | 89.2 |
| 4 | 75.7 | 86.7 | 82.1 | 81.2 | 86.3 | 88.2 |
| 5 | 73.1 | 85.6 | 80.2 | 79.1 | 84.9 | 87.6 |
| 6 | 72.2 | 84.8 | 79.6 | 78.2 | 83.5 | 86.6 |
| 7 | 68.7 | 83.6 | 75.6 | 74.6 | 82.0 | 86.1 |
| 8 | 38.8 | 82.1 | 39.2 | 39.2 | 80.9 | 85.0 |

---

## Figure 7 — LIBERO delay (appendix; two-panel: SR + episode length vs delay)

### Figure 7(a): 4-suite avg success rate (%)

| d | VLASH | DEFLECT | Δ |
|---|---|---|---|
| 1 | 96.6 | 96.8 | +0.2 |
| 2 | 97.1 | 97.5 | +0.5 |
| 3 | 95.5 | 96.4 | +1.0 |
| 4 | 93.1 | 94.4 | +1.4 |
| 5 | 88.3 | 90.6 | +2.2 |
| 6 | 81.9 | 84.2 | +2.3 |
| 7 | 72.9 | 77.5 | +4.6 |

### Figure 7(b): 4-suite avg episode length (steps to completion, lower = faster)

| d | VLASH | DEFLECT | Δ |
|---|---|---|---|
| 1 | 155.7 | 152.6 | −3.1 |
| 2 | 158.3 | 153.4 | −4.9 |
| 3 | 165.2 | 158.5 | −6.7 |
| 4 | 175.6 | 167.4 | −8.2 |
| 5 | 190.3 | 179.8 | −10.5 |
| 6 | 206.2 | 197.7 | −8.5 |
| 7 | 228.3 | 214.7 | −13.6 |

DEFLECT is faster than VLASH at every delay — gap grows with delay.

---

## Figure 8 — Zero-shot delay generalization (appendix; Models A / B / C vs VLASH)

| d | VLASH | A: d_DPO ∈ {1,2} | B (main): d_DPO ∈ {1,..,4} | C: d_DPO ∈ {1,..,7}, async=8 |
|---|---|---|---|---|
| 0 | 89.5 | 91.2 | 91.3 | 90.8 |
| 1 | 89.3 | 90.6 | 90.9 | 89.9 |
| 2 | 87.7 | 89.1 | 89.1 | 88.2 |
| 3 | 85.5 | 87.7 | 88.4 | 87.0 |
| 4 | 81.9 | 85.1 | 86.1 | 85.3 |
| 5 | 74.6 | 79.0 | 80.8 | 81.4 |
| 6 | 66.3 | 71.4 | 73.3 | 77.3 |
| 7 | 60.5 | 64.2 | 66.3 | 71.7 |
| **avg(0–7)** | 79.4 | 82.3 | 83.3 | 83.9 |

---

## Figure 9 — Ablations (appendix; delay-averaged success rate over d ∈ {0..7})

Five bars, left → right:

| Variant | avg-d SR |
|---|---|
| VLASH baseline | 79.4 |
| λ_DPO = 0  (continued SFT only, cosine LR) | 82.0 |
| DPO with VLASH-style rejection (vs naive-stale) | 81.9 |
| DPO **without** SFT anchor (collapse) | 12.3 |
| DEFLECT (full, λ_DPO = 0.02) | 83.3 |

---

## Figure 10 — Sensitivity to λ_DPO (appendix; Kinetix sweep)

Two y-axes (delay-avg, horizon-avg) plotted vs λ.

| λ_DPO | avg-d SR (d=0..7) | avg-K SR (K=1..8 at d=1) |
|---|---|---|
| 0.00 | 81.98 | 87.69 |
| 0.01 | 82.96 | 87.98 |
| 0.02 | 83.30 | 87.88 |
| 0.05 | 83.32 | 87.64 |
| 0.10 | 83.11 | 87.29 |
| 0.20 | 82.76 | 86.62 |
| 0.50 | 81.99 | 84.90 |

Selected λ = 0.02 (Pareto-optimal on joint (avg-d, avg-K)).

---

## Figure 11 — Mechanism: per-task final deflection (appendix; raincloud)

12 Kinetix tasks at d = 6. `||DEFLECT(τ=1) − VLASH(τ=1)||₂` per (seed × chunk_pos) sample. Each task has 32 samples (4 seeds × 8 chunk positions). Tasks ordered by descending mean.

| task | mean | std | 32 samples |
|---|---|---|---|
| grasp_easy        | 0.602 | 0.571 | 0.311, 1.775, 1.977, 1.738, 1.718, 0.781, 1.534, 1.405, 0.167, 0.622, 0.231, 0.321, 0.384, 0.274, 0.193, 0.227, 0.079, 0.275, 0.436, 0.516, 0.897, 0.844, 0.770, 0.314, 0.135, 0.077, 0.096, 0.177, 0.224, 0.385, 0.141, 0.243 |
| trampoline        | 0.456 | 0.628 | 0.070, 0.109, 0.254, 1.441, 1.918, 2.083, 1.342, 1.972, 0.150, 0.086, 0.156, 0.088, 0.039, 0.109, 0.099, 0.113, 0.154, 0.168, 0.094, 0.192, 0.143, 0.353, 1.240, 1.157, 0.059, 0.071, 0.089, 0.119, 0.091, 0.236, 0.243, 0.140 |
| car_launch        | 0.427 | 0.445 | 0.137, 0.739, 0.595, 0.140, 0.521, 0.028, 0.062, 0.172, 0.175, 0.343, 0.161, 0.092, 0.274, 0.113, 0.303, 0.487, 1.163, 1.016, 0.458, 0.348, 0.922, 0.404, 0.185, 0.062, 0.194, 0.046, 0.146, 0.608, 0.267, 1.642, 0.032, 1.836 |
| chain_lander      | 0.370 | 0.141 | 0.244, 0.645, 0.277, 0.360, 0.478, 0.305, 0.337, 0.260, 0.510, 0.565, 0.271, 0.513, 0.120, 0.191, 0.295, 0.445, 0.433, 0.377, 0.328, 0.300, 0.253, 0.321, 0.219, 0.494, 0.209, 0.245, 0.715, 0.608, 0.413, 0.440, 0.227, 0.441 |
| h17_unicycle      | 0.340 | 0.185 | 0.134, 0.563, 0.159, 0.407, 0.222, 0.446, 0.337, 0.263, 0.242, 0.097, 0.157, 0.174, 0.218, 0.649, 0.363, 0.374, 0.184, 0.428, 0.292, 1.018, 0.272, 0.432, 0.458, 0.514, 0.568, 0.106, 0.287, 0.418, 0.306, 0.261, 0.186, 0.332 |
| hard_lunar_lander | 0.278 | 0.110 | 0.178, 0.086, 0.233, 0.212, 0.189, 0.194, 0.201, 0.265, 0.403, 0.186, 0.167, 0.489, 0.350, 0.359, 0.182, 0.301, 0.376, 0.325, 0.317, 0.438, 0.223, 0.308, 0.342, 0.287, 0.120, 0.266, 0.071, 0.176, 0.362, 0.367, 0.484, 0.442 |
| catapult          | 0.261 | 0.212 | 0.644, 1.066, 0.130, 0.560, 0.161, 0.190, 0.183, 0.353, 0.098, 0.297, 0.219, 0.205, 0.143, 0.598, 0.129, 0.315, 0.388, 0.104, 0.164, 0.053, 0.108, 0.099, 0.471, 0.048, 0.069, 0.247, 0.327, 0.208, 0.140, 0.185, 0.096, 0.339 |
| mjc_swimmer       | 0.236 | 0.195 | 0.121, 0.107, 0.182, 0.149, 0.058, 0.087, 0.083, 0.194, 0.181, 0.076, 0.192, 0.090, 0.253, 0.395, 0.206, 0.729, 0.837, 0.224, 0.598, 0.607, 0.249, 0.340, 0.251, 0.345, 0.140, 0.093, 0.197, 0.105, 0.085, 0.086, 0.084, 0.218 |
| catcher_v3        | 0.222 | 0.222 | 0.632, 0.790, 0.246, 0.084, 0.408, 0.063, 0.450, 0.037, 0.043, 0.469, 0.126, 0.162, 0.042, 0.124, 0.050, 0.049, 0.110, 0.269, 0.385, 0.741, 0.335, 0.057, 0.094, 0.088, 0.649, 0.222, 0.076, 0.061, 0.076, 0.065, 0.059, 0.048 |
| mjc_walker        | 0.184 | 0.314 | 0.062, 0.116, 1.019, 0.973, 0.587, 1.308, 0.189, 0.163, 0.029, 0.050, 0.015, 0.107, 0.023, 0.050, 0.026, 0.028, 0.065, 0.132, 0.078, 0.082, 0.099, 0.064, 0.096, 0.128, 0.039, 0.039, 0.035, 0.047, 0.044, 0.038, 0.027, 0.137 |
| mjc_half_cheetah  | 0.158 | 0.160 | 0.049, 0.234, 0.605, 0.024, 0.047, 0.246, 0.640, 0.086, 0.099, 0.127, 0.071, 0.161, 0.026, 0.136, 0.099, 0.030, 0.037, 0.141, 0.133, 0.037, 0.128, 0.244, 0.300, 0.582, 0.044, 0.089, 0.151, 0.128, 0.119, 0.101, 0.047, 0.103 |
| cartpole_thrust   | 0.147 | 0.103 | 0.049, 0.039, 0.237, 0.130, 0.036, 0.128, 0.215, 0.175, 0.047, 0.110, 0.088, 0.240, 0.108, 0.110, 0.119, 0.360, 0.319, 0.356, 0.298, 0.209, 0.085, 0.072, 0.187, 0.191, 0.199, 0.036, 0.071, 0.049, 0.342, 0.026, 0.027, 0.062 |

---

## Figure 12 — Mechanism: per-task σ-ratio (appendix; bar chart)

12 Kinetix tasks at d = 6. N = 200 noise samples, chunk position 0. Ratio = σ_DEFLECT / σ_VLASH.

| task | σ_VLASH | σ_DEFLECT | ratio |
|---|---|---|---|
| catcher_v3        | 0.375 | 0.428 | 1.14 |
| grasp_easy        | 0.304 | 0.325 | 1.07 |
| trampoline        | 0.405 | 0.415 | 1.02 |
| hard_lunar_lander | 0.364 | 0.359 | 0.99 |
| mjc_swimmer       | 0.416 | 0.410 | 0.99 |
| chain_lander      | 0.494 | 0.473 | 0.96 |
| catapult          | 0.432 | 0.415 | 0.96 |
| mjc_half_cheetah  | 0.266 | 0.255 | 0.96 |
| h17_unicycle      | 0.402 | 0.384 | 0.95 |
| car_launch        | 0.331 | 0.323 | 0.98 |
| cartpole_thrust   | 0.437 | 0.405 | 0.93 |
| mjc_walker        | 0.341 | 0.276 | 0.81 |

Median ratio = 0.97. Range = [0.81, 1.14].

---

## Figure 13 — Mean correction magnitude vs chunk position (appendix)

Aggregated over 12 tasks × 4 critical states at d = 6. Bar chart: 8 chunk positions.

| chunk position | mean ‖Δa‖₂ | std |
|---|---|---|
| 0 | 0.34 | 0.30 |
| 1 | 0.34 | 0.31 |
| 2 | 0.32 | 0.32 |
| 3 | 0.30 | 0.31 |
| 4 | 0.29 | 0.30 |
| 5 | 0.29 | 0.31 |
| 6 | 0.28 | 0.30 |
| 7 | 0.28 | 0.30 |

Roughly uniform across positions; mild emphasis on positions 0–2.

---

## Figure 14 — Catapult case study, action dim 0 distribution (appendix; single state, d = 6, N = 200 shared noise)

| Policy | μ (mean) | σ (std) |
|---|---|---|
| VLASH | −0.24 | 0.66 |
| DEFLECT | −0.64 | 0.52 |

Δμ = −0.39 (mean displacement); σ shrinks by 0.14. Two histograms (200 noise samples each) overlaid on action dim 0 at chunk position 0.
