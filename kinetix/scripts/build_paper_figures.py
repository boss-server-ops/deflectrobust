"""Build CoRL paper figures from raw eval data + audit numbers.

Run with: cd .../benchmarks/kinetix && uv run python scripts/build_paper_figures.py
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

K_ROOT = pathlib.Path("<repo_root>")
L_ROOT = pathlib.Path("<libero_repo>/results")
OUT = pathlib.Path("<paper_figures>")
OUT.mkdir(parents=True, exist_ok=True)

DELAY_PAIRS = [(d, max(1, d)) for d in range(8)]
HORIZON_PAIRS = [(1, K) for K in range(1, 9)]


# ---------- Kinetix data loaders ----------

def kx_load_csv(path):
    df = pd.read_csv(path)
    df["sr"] = df["returned_episode_solved"] * 100
    return df


def kx_delay(df, method):
    sub = df[df["method"] == method] if "method" in df.columns else df
    out = []
    for d, h in DELAY_PAIRS:
        m = sub[(sub["delay"] == d) & (sub["execute_horizon"] == h)]
        out.append((d, m["sr"].mean() if len(m) else np.nan))
    return pd.Series(dict(out))


def kx_horizon(df, method):
    sub = df[df["method"] == method] if "method" in df.columns else df
    out = []
    for d, h in HORIZON_PAIRS:
        m = sub[(sub["delay"] == d) & (sub["execute_horizon"] == h)]
        out.append((h, m["sr"].mean() if len(m) else np.nan))
    return pd.Series(dict(out))


def dpo_run_delay(run_dir):
    p = K_ROOT / "train_outputs" / run_dir
    cands = sorted(p.glob("eval_ep*_delay/results_delay.csv"))
    if not cands:
        return None
    df = kx_load_csv(cands[-1])  # last (highest ep)
    return kx_delay(df, "vlash"), cands[-1]


def dpo_run_horizon(run_dir):
    p = K_ROOT / "train_outputs" / run_dir
    cands = sorted(p.glob("eval_ep*_horizon/results_horizon.csv"))
    if not cands:
        return None
    df = kx_load_csv(cands[-1])
    return kx_horizon(df, "vlash"), cands[-1]


# ---------- LIBERO loader ----------

def lib_load(method_dir, suite, delay):
    p = L_ROOT / method_dir / f"libero_{suite}" / f"d{delay}.json"
    if not p.exists():
        return np.nan
    with p.open() as f:
        d = json.load(f)
    return d["overall"]["pc_successes"]


# ============================================================
# AUDIT — print computed values
# ============================================================

print("=" * 70)
print("AUDIT — values computed from raw CSVs / JSONs")
print("=" * 70)

# Baselines (Naive, VLASH, RTC, BID)
base_csv = K_ROOT / "eval_outputs_fig6/20260407_174901/results.csv"
rtc_csv = K_ROOT / "eval_outputs_fig6/269650/results_delay.csv"
rtc_hcsv = K_ROOT / "eval_outputs_fig6/269650/results_horizon.csv"
bid_csv = K_ROOT / "eval_outputs_fig6/269657/results_delay.csv"
bid_hcsv = K_ROOT / "eval_outputs_fig6/269657/results_horizon.csv"

base_df = kx_load_csv(base_csv)
rtc_d = kx_delay(kx_load_csv(rtc_csv), "realtime")
rtc_h = kx_horizon(kx_load_csv(rtc_hcsv), "realtime")
bid_d = kx_delay(kx_load_csv(bid_csv), "bid")
bid_h = kx_horizon(kx_load_csv(bid_hcsv), "bid")

naive_d = kx_delay(base_df, "naive")
vlash_d = kx_delay(base_df, "vlash")
naive_h = kx_horizon(base_df, "naive")
vlash_h = kx_horizon(base_df, "vlash")

# DEFLECT main line: vanilla DPO, λ=0.02 (best on horizon avg, ties λ=0.05 on delay avg).
DPO_MAIN_RUN = "dpo_vanilla_ep24_cosine_d0_lam0.02_269992"
DPO_MAIN_LAMBDA = 0.02
dpo_d, dpo_d_src = dpo_run_delay(DPO_MAIN_RUN)
dpo_h, dpo_h_src = dpo_run_horizon(DPO_MAIN_RUN)

print("\nKINETIX DELAY (12-task avg, success%)")
print(f"{'d':>3} | {'Naive':>6} {'VLASH':>6} {'RTC':>6} {'BID':>6} {'DPO':>6}")
for d in range(8):
    print(f"{d:>3} | {naive_d[d]:>6.1f} {vlash_d[d]:>6.1f} {rtc_d[d]:>6.1f} {bid_d[d]:>6.1f} {dpo_d[d]:>6.1f}")
print(f"avg | {naive_d.mean():>6.1f} {vlash_d.mean():>6.1f} {rtc_d.mean():>6.1f} {bid_d.mean():>6.1f} {dpo_d.mean():>6.1f}")

print("\nKINETIX HORIZON (at d=1, 12-task avg, success%)")
print(f"{'K':>3} | {'Naive':>6} {'VLASH':>6} {'RTC':>6} {'BID':>6} {'DPO':>6}")
for K in range(1, 9):
    print(f"{K:>3} | {naive_h[K]:>6.1f} {vlash_h[K]:>6.1f} {rtc_h[K]:>6.1f} {bid_h[K]:>6.1f} {dpo_h[K]:>6.1f}")
print(f"avg | {naive_h.mean():>6.1f} {vlash_h.mean():>6.1f} {rtc_h.mean():>6.1f} {bid_h.mean():>6.1f} {dpo_h.mean():>6.1f}")

# Lambda sweep — vanilla DPO only (γ=0). LAM-ON runs (268906, 268907) excluded.
# 0.0 = pure SFT with cosine LR (job 268956, γ=1.0 with λ=0 → γ is moot since
# DPO term is zero-weighted; functionally identical to vanilla SFT cosine).
# λ=0.01 and λ=0.02 placeholders: jobs 269707/269708 still training.
LAMBDA_RUNS = {
    0.0:  "lam_g1.0_ep24_cosine_d0_lam0_268956",   # pure SFT cosine (λ=0)
    0.01: "dpo_vanilla_ep24_cosine_d0_lam0.01_269707",
    0.02: "dpo_vanilla_ep24_cosine_d0_lam0.02_269992",
    0.05: "lam_g0_ep24_cosine_d0_lam0.05_268710",
    0.1:  "lam_g0_ep24_cosine_d0_268393",           # vanilla γ=0, λ=0.1
    0.2:  "lam_g0_ep24_cosine_d0_lam0.2_268711",
    0.5:  "lam_g0_ep24_cosine_d0_lam0.5_268712",
}
print("\nKINETIX LAMBDA SWEEP (vanilla DPO γ=0, avg over d=0..7 / K=1..8)")
print(f"{'lam':>6} | {'avg-d':>6} {'avg-K':>6}  source-d")
lam_d, lam_h = {}, {}
for lam, run in LAMBDA_RUNS.items():
    if run is None:
        lam_d[lam] = None
        lam_h[lam] = None
        print(f"{lam:>6} | {'PEND':>6} {'PEND':>6}  (training in progress)")
        continue
    rd = dpo_run_delay(run)
    rh = dpo_run_horizon(run)
    avg_d = rd[0].mean() if rd else np.nan
    avg_h = rh[0].mean() if rh else np.nan
    lam_d[lam] = rd[0] if rd else None
    lam_h[lam] = rh[0] if rh else None
    src = rd[1].relative_to(K_ROOT) if rd else "MISSING"
    print(f"{lam:>6} | {avg_d:>6.1f} {avg_h:>6.1f}  {src}")

# Ablations: keep ONE variable at a time. Both ablation runs (268708, 268709)
# were trained at λ=0.1 with γ=0; the matched control (naive reject + SFT anchor)
# is therefore 268393 (vanilla γ=0, λ=0.1), not the main λ=0.05 run.
abl_ctrl_d        = dpo_run_delay("lam_g0_ep24_cosine_d0_268393")[0]   # λ=0.1 naive + anchor
abl_rej_vlash_d   = dpo_run_delay("lam_g0_ep24_cosine_d0_rvlash_268708")[0]
abl_anchor_off_d  = dpo_run_delay("lam_g0_ep24_cosine_d0_sft0_268709")[0]
abl_lam0_d        = dpo_run_delay("lam_g1.0_ep24_cosine_d0_lam0_268956")[0]  # true λ=0 (pure SFT cosine)

print("\nKINETIX ABLATIONS (avg over d=0..7, all γ=0)")
print(f"  vlash baseline:                          {vlash_d.mean():.1f}")
print(f"  λ=0 (pure SFT cosine — LR artifact):     {abl_lam0_d.mean():.1f}")
print(f"  λ=0.1 + naive reject + anchor (control): {abl_ctrl_d.mean():.1f}")
print(f"  λ=0.1 + VLASH-style reject + anchor:     {abl_rej_vlash_d.mean():.1f}")
print(f"  λ=0.1 + naive reject + NO anchor:        {abl_anchor_off_d.mean():.1f}")
print(f"  DEFLECT main (λ=0.05 vanilla):          {dpo_d.mean():.1f}")

# LIBERO 4-suite avg
SUITES = ["spatial", "object", "goal", "10"]
print("\nLIBERO 4-suite avg (success%)")
print(f"{'d':>3} | {'VLASH':>6} {'DPO':>6}  missing-files-flag")
lib_v_avg, lib_d_avg = {}, {}
for d in range(1, 8):
    v = [lib_load("vlash_baseline", s, d) for s in SUITES]
    p = [lib_load("dpo_naive_200", s, d) for s in SUITES]
    miss_v = sum(np.isnan(v))
    miss_p = sum(np.isnan(p))
    flag = ""
    if miss_v: flag += f" VLASH-miss={miss_v}"
    if miss_p: flag += f" DPO-miss={miss_p}"
    lib_v_avg[d] = np.nanmean(v)
    lib_d_avg[d] = np.nanmean(p)
    print(f"{d:>3} | {lib_v_avg[d]:>6.1f} {lib_d_avg[d]:>6.1f} {flag}")

# ============================================================
# FIGURES
# ============================================================

print("\n" + "=" * 70)
print("Generating PDF figures …")
print("=" * 70)

COLORS = {
    "Naive":    "#888888",
    "VLASH":    "#0b3d91",
    "RTC":      "#1b9e77",
    "BID":      "#9c52a0",
    "PFM":      "#c0392b",
    "DEFLECT": "#d97706",
}
MARKERS = {"Naive": "v", "VLASH": "o", "RTC": "s", "BID": "D", "PFM": "P", "DEFLECT": "^"}

# PFM σ=0 delay results (clean-obs preference baseline, no delay in preferences)
_pfm_csv = pathlib.Path("<pfm_repo>/eval_results/pfm/results_delay.csv")
if _pfm_csv.exists():
    _pfm_df = pd.read_csv(_pfm_csv)
    # This CSV stores pc_successes already in percent (not 0-1)
    pfm_d = _pfm_df.groupby("delay")["returned_episode_solved"].mean()
else:
    pfm_d = None
# PFM horizon (d=1, K=1..8)
_pfm_h_csv = pathlib.Path("<pfm_repo>/eval_results/pfm/results_horizon.csv")
if _pfm_h_csv.exists():
    _pfm_h_df = pd.read_csv(_pfm_h_csv)
    pfm_h = _pfm_h_df[_pfm_h_df["delay"] == 1].groupby("execute_horizon")["returned_episode_solved"].mean()
else:
    pfm_h = None


def plot_curves(ax, x, series_dict, xlabel, ylabel):
    for name, y in series_dict.items():
        ax.plot(x, y, marker=MARKERS[name], color=COLORS[name],
                linewidth=2, markersize=6, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)


# F1 — Kinetix delay (with PFM preference-signal baseline)
fig, ax = plt.subplots(figsize=(8.1, 5.6))
_series = {
    "Naive": naive_d.values, "VLASH": vlash_d.values,
    "RTC": rtc_d.values, "BID": bid_d.values,
}
if pfm_d is not None:
    _series["PFM"] = pfm_d.values
_series["DEFLECT"] = dpo_d.values
plot_curves(ax, list(range(8)), _series,
            "Inference delay $\\Delta$", "Success rate (%)")
ax.set_xticks(range(8))
plt.tight_layout()
plt.savefig(OUT / "F1_kinetix_delay.pdf", bbox_inches="tight")
plt.close()
print("  F1_kinetix_delay.pdf")

# F2 — Kinetix horizon (with PFM line when available)
fig, ax = plt.subplots(figsize=(8.1, 5.6))
_h_series = {
    "Naive": naive_h.values, "VLASH": vlash_h.values,
    "RTC": rtc_h.values, "BID": bid_h.values,
}
if pfm_h is not None:
    _h_series["PFM"] = pfm_h.values
_h_series["DEFLECT"] = dpo_h.values
plot_curves(ax, list(range(1, 9)), _h_series,
            "Execution horizon $K$", "Success rate (%)")
ax.set_xticks(range(1, 9))
plt.tight_layout()
plt.savefig(OUT / "F2_kinetix_horizon.pdf", bbox_inches="tight")
plt.close()
print("  F2_kinetix_horizon.pdf")

# F3 — Lambda sweep (vanilla DPO γ=0). PEND lambdas drawn as hatched gray bars.
fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.0))
lams_sorted = sorted(LAMBDA_RUNS.keys())
avg_d_vals = [lam_d[l].mean() if lam_d[l] is not None else np.nan for l in lams_sorted]
avg_h_vals = [lam_h[l].mean() if lam_h[l] is not None else np.nan for l in lams_sorted]
xs = np.arange(len(lams_sorted))

def _plot_sweep(ax, vals, baseline, ylabel):
    for x, v in zip(xs, vals):
        if np.isnan(v):
            ax.bar(x, baseline * 0.95, color="#dddddd", edgecolor="#888888",
                   linewidth=0.6, hatch="///", alpha=0.6)
            ax.text(x, baseline * 0.95 + 0.5, "pending", ha="center",
                    fontsize=11, style="italic", color="#555555")
        else:
            ax.bar(x, v, color="#d97706", edgecolor="black", linewidth=0.5)
            ax.text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=12)
    filled = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
    if filled:
        best_i = max(filled, key=lambda t: t[1])[0]
        ax.bar(xs[best_i], vals[best_i], color="#7a3600", edgecolor="black", linewidth=1.0)
    ax.axhline(baseline, color="#0b3d91", linestyle="--", linewidth=1.5,
               label="VLASH")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l:g}" for l in lams_sorted])
    ax.set_xlabel("$\\lambda_{\\mathrm{DPO}}$")
    ax.set_ylabel(ylabel)
    ymax = max([baseline] + [v for v in vals if not np.isnan(v)]) * 1.12
    ax.set_ylim(0, ymax)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

_plot_sweep(axes[0], avg_d_vals, vlash_d.mean(),
            "Avg success rate (%)\n$\\Delta$=0–7")
_plot_sweep(axes[1], avg_h_vals, vlash_h.mean(),
            "Avg success rate (%)\n$K$=1–8")

plt.tight_layout()
plt.savefig(OUT / "F3_lambda_sweep.pdf", bbox_inches="tight")
plt.close()
print("  F3_lambda_sweep.pdf")

# F4 — Ablations bar chart
fig, ax = plt.subplots(figsize=(10.5, 5.9))
labels = [
    "VLASH\n(baseline)",
    "$\\lambda$=0\n(SFT only,\ncosine LR)",
    "DPO\nVLASH-style\nrejection",
    "DPO\nno SFT\nanchor",
    "DEFLECT\n(ours,\n$\\lambda$=0.02)",
]
vals = [
    vlash_d.mean(),
    abl_lam0_d.mean(),
    abl_rej_vlash_d.mean(),
    abl_anchor_off_d.mean(),
    dpo_d.mean(),
]
colors = ["#0b3d91", "#aaaaaa", "#cc8888", "#883333", "#d97706"]
xs = np.arange(len(labels))
bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.6)
for x, v in zip(xs, vals):
    ax.text(x, v + 1.0 if v > 0 else v + 1.5, f"{v:.1f}",
            ha="center", fontsize=13, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylabel("Avg success rate (%)\n$\\Delta$=0–7")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(vals) * 1.12)
plt.tight_layout()
plt.savefig(OUT / "F4_ablations.pdf", bbox_inches="tight")
plt.close()
print("  F4_ablations.pdf")

# F5 — LIBERO delay
fig, ax = plt.subplots(figsize=(7.7, 5.6))
ds = list(range(1, 8))
ax.plot(ds, [lib_v_avg[d] for d in ds], "o-", color=COLORS["VLASH"],
        linewidth=2, markersize=6, label="VLASH")
ax.plot(ds, [lib_d_avg[d] for d in ds], "^-", color=COLORS["DEFLECT"],
        linewidth=2, markersize=6, label="DEFLECT")
ax.set_xlabel("Inference delay $\\Delta$")
ax.set_ylabel("Success rate (%)")
ax.set_xticks(ds)
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT / "F5_libero_delay.pdf", bbox_inches="tight")
plt.close()
print("  F5_libero_delay.pdf")

# F6 — Qualitative: ODE + distribution analysis (catapult d=6) — 6 separate PDFs
ode = np.load(K_ROOT / "figures/ode/worlds_l_catapult_d6_ode_data.npz")
dist = np.load(K_ROOT / "figures/distribution/worlds_l_catapult_d6_dist_data.npz")
v = ode["vlash_states"]; o = ode["dpo_states"]; t = ode["times"]
v_pts = dist["vlash_actions"][:, 0]  # (N, D)
d_pts = dist["dpo_actions"][:, 0]

# F6a: deflection vs ODE time
fig, ax = plt.subplots(figsize=(5.9, 4.8))
ax.plot(ode["times"], ode["delta_norm"], "o-", color="#d97706", linewidth=2, markersize=5)
ax.fill_between(ode["times"], 0, ode["delta_norm"], alpha=0.2, color="#d97706")
ax.set_xlabel("ODE time $\\tau$")
ax.set_ylabel(r"$\|a_{\mathrm{DEFLECT}}(\tau) - a_{\mathrm{VLASH}}(\tau)\|_2$")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "F6a_ode_deflection.pdf", bbox_inches="tight")
plt.close()
print("  F6a_ode_deflection.pdf")

# F6b: per-dim ODE evolution (dim 0)
fig, ax = plt.subplots(figsize=(5.9, 4.8))
ax.plot(t, v[:, 0, 0], "o-", color=COLORS["VLASH"], linewidth=2, markersize=4, label="VLASH")
ax.plot(t, o[:, 0, 0], "^-", color=COLORS["DEFLECT"], linewidth=2, markersize=4, label="DEFLECT")
ax.set_xlabel("ODE time $\\tau$")
ax.set_ylabel("Action dim 0 (chunk pos 0)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT / "F6b_ode_per_dim.pdf", bbox_inches="tight")
plt.close()
print("  F6b_ode_per_dim.pdf")

# F6c: heatmap (tau × chunk pos)
fig, ax = plt.subplots(figsize=(6.4, 4.8))
dn_full = ode["delta_norm_full"].T  # (chunk_pos, tau)
im = ax.imshow(dn_full, aspect="auto", origin="lower",
               extent=[t[0], t[-1], -0.5, dn_full.shape[0] - 0.5], cmap="Oranges")
ax.set_xlabel("ODE time $\\tau$")
ax.set_ylabel("Action chunk position")
plt.colorbar(im, ax=ax, label=r"$\|\Delta a\|_2$")
plt.tight_layout()
plt.savefig(OUT / "F6c_ode_heatmap.pdf", bbox_inches="tight")
plt.close()
print("  F6c_ode_heatmap.pdf")

# F6d: 2D scatter (dim 0, dim 1)
fig, ax = plt.subplots(figsize=(5.9, 5.0))
ax.scatter(v_pts[:, 0], v_pts[:, 1], s=14, c=COLORS["VLASH"], alpha=0.45, edgecolors="none", label="VLASH")
ax.scatter(d_pts[:, 0], d_pts[:, 1], s=14, c=COLORS["DEFLECT"], alpha=0.45, edgecolors="none", label="DEFLECT")
ax.scatter(*v_pts[:, :2].mean(axis=0), s=180, c="white", edgecolors=COLORS["VLASH"],
           linewidths=2.5, marker="X", zorder=5)
ax.scatter(*d_pts[:, :2].mean(axis=0), s=180, c="white", edgecolors=COLORS["DEFLECT"],
           linewidths=2.5, marker="X", zorder=5)
ax.set_xlabel("Action dim 0")
ax.set_ylabel("Action dim 1")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT / "F6d_dist_scatter.pdf", bbox_inches="tight")
plt.close()
print("  F6d_dist_scatter.pdf")

# F6e: dim 0 histogram (the changed dim)
fig, ax = plt.subplots(figsize=(5.9, 4.8))
ax.hist(v_pts[:, 0], bins=30, color=COLORS["VLASH"], alpha=0.55, density=True, label="VLASH")
ax.hist(d_pts[:, 0], bins=30, color=COLORS["DEFLECT"], alpha=0.55, density=True, label="DEFLECT")
ax.axvline(v_pts[:, 0].mean(), color=COLORS["VLASH"], linestyle="--", linewidth=2)
ax.axvline(d_pts[:, 0].mean(), color=COLORS["DEFLECT"], linestyle="--", linewidth=2)
ax.set_xlabel("Action dim 0 value")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT / "F6e_dist_hist_dim0.pdf", bbox_inches="tight")
plt.close()
print("  F6e_dist_hist_dim0.pdf")

# F6f: dim 1 histogram (orthogonal/unchanged dim)
fig, ax = plt.subplots(figsize=(5.9, 4.8))
ax.hist(v_pts[:, 1], bins=30, color=COLORS["VLASH"], alpha=0.55, density=True, label="VLASH")
ax.hist(d_pts[:, 1], bins=30, color=COLORS["DEFLECT"], alpha=0.55, density=True, label="DEFLECT")
ax.axvline(v_pts[:, 1].mean(), color=COLORS["VLASH"], linestyle="--", linewidth=2)
ax.axvline(d_pts[:, 1].mean(), color=COLORS["DEFLECT"], linestyle="--", linewidth=2)
ax.set_xlabel("Action dim 1 value")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT / "F6f_dist_hist_dim1.pdf", bbox_inches="tight")
plt.close()
print("  F6f_dist_hist_dim1.pdf")

# F7 — Multi-task deflection-energy statistics (defensive against cherry-pick)
# Split into 3 separate PDFs for flexible paper layout.
agg_npz = K_ROOT / "figures/aggregate/aggregate_ode_lam0.05.npz"
if agg_npz.exists():
    ag = np.load(agg_npz, allow_pickle=True)
    task_names_raw = ag["task_names"]
    # Concat all per-task (seeds, T+1, chunk) into one big (N, T+1, chunk)
    stack = np.concatenate(
        [ag[f"dnf_{i}"] for i in range(len(task_names_raw))], axis=0
    )  # shape (sum_seeds, T+1, chunk)
    times_tau = np.linspace(0, 1, stack.shape[1])
    chunk_positions = np.arange(stack.shape[2])

    # --- F7a: late-stage τ injection (the strong universal claim) ---
    per_tau = stack.mean(axis=-1)  # (N, T+1)
    mean_tau = per_tau.mean(axis=0)
    std_tau  = per_tau.std(axis=0)
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    ax.plot(times_tau, mean_tau, "o-", color="#d97706", linewidth=2, markersize=5)
    ax.fill_between(times_tau, mean_tau - std_tau, mean_tau + std_tau,
                    color="#d97706", alpha=0.22, linewidth=0)
    ax.set_xlabel(r"ODE time $\tau$")
    ax.set_ylabel(r"Deflection $\|\Delta a\|_2$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "F7a_tau_evolution.pdf", bbox_inches="tight")
    plt.close()
    print(f"  F7a_tau_evolution.pdf  "
          f"(aggregated over {len(task_names_raw)} tasks × "
          f"{stack.shape[0] // len(task_names_raw)} seeds)")

    # --- F7b: chunk-position bars (broadly uniform — kept for transparency) ---
    final = stack[:, -1, :]  # (N, chunk)
    mean_pos = final.mean(axis=0)
    std_pos  = final.std(axis=0)
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    ax.bar(chunk_positions, mean_pos, yerr=std_pos, color="#d97706",
           edgecolor="black", linewidth=0.5, capsize=3,
           error_kw={"ecolor": "#555555", "lw": 1})
    ax.set_xticks(chunk_positions)
    ax.set_xlabel("Action chunk position")
    ax.set_ylabel(r"Final deflection $\|\Delta a(\tau{=}1)\|_2$")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "F7b_chunk_position.pdf", bbox_inches="tight")
    plt.close()
    print("  F7b_chunk_position.pdf")

    # --- F7c: per-task raincloud plot (half-violin + boxplot + strip) ---
    # Each task: 4 seeds × 8 chunk positions = 32 sample points at τ=1.
    # Shows the FULL distribution (not just mean±std) for reviewer rigor.
    task_samples, task_labels = [], []
    for i, full in enumerate(task_names_raw):
        d = ag[f"dnf_{i}"]              # (seeds, T+1, chunk)
        samples = d[:, -1, :].reshape(-1)  # 32 scalar samples at final τ
        task_samples.append(samples)
        task_labels.append(str(full).replace("worlds/l/", "").replace(".json", ""))

    order = np.argsort([-s.mean() for s in task_samples])  # descending mean
    sorted_samples = [task_samples[i] for i in order]
    sorted_labels  = [task_labels[i]  for i in order]

    fig, ax = plt.subplots(figsize=(10.5, 7.8))
    color = "#d97706"
    ys = np.arange(len(sorted_samples))

    # Half-violins (upper half only)
    parts = ax.violinplot(sorted_samples, positions=ys, vert=False,
                          widths=0.82, showmeans=False, showmedians=False,
                          showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        # keep only upper half (y > task_center)
        y0 = ys[i]
        verts = body.get_paths()[0].vertices
        verts[:, 1] = np.clip(verts[:, 1], y0, None)
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor("black")
        body.set_linewidth(0.4)

    # Horizontal boxplots on the task axis
    bp = ax.boxplot(sorted_samples, positions=ys, vert=False,
                    widths=0.12, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2),
                    boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
                    whiskerprops=dict(color="black", linewidth=0.8),
                    capprops=dict(color="black", linewidth=0.8))

    # Strip plot below boxplot (individual sample dots)
    rng = np.random.default_rng(0)
    for i, samples in enumerate(sorted_samples):
        jitter = rng.normal(0, 0.05, size=len(samples))
        ax.scatter(samples, np.full_like(samples, ys[i] - 0.22) + jitter,
                   s=10, c=color, alpha=0.55, edgecolors="none")

    ax.set_yticks(ys)
    ax.set_yticklabels(sorted_labels)
    ax.invert_yaxis()  # highest-deflection task on top
    ax.set_xlabel(r"Final deflection $\|\Delta a(\tau{=}1)\|_2$  (per seed × chunk pos)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT / "F7c_per_task_deflection.pdf", bbox_inches="tight")
    plt.close()
    total_n = sum(len(s) for s in sorted_samples)
    print(f"  F7c_per_task_deflection.pdf  "
          f"(raincloud: {len(sorted_samples)} tasks, "
          f"{total_n // len(sorted_samples)} samples each, {total_n} total)")
else:
    print("  F7 skipped: aggregate_ode npz not yet available "
          f"({agg_npz})")

# F8 — Multi-task action-distribution variance comparison.
# Tests whether the catapult observation (σ_DEFLECT < σ_VLASH at chunk pos 0)
# generalizes across all 12 Kinetix tasks.
dist_npz = K_ROOT / "figures/aggregate/distribution_across_tasks_lam0.02.npz"
if dist_npz.exists():
    dn = np.load(dist_npz, allow_pickle=True)
    task_names_d = dn["task_names"]
    rows = []
    for i, t in enumerate(task_names_d):
        v = dn[f"vlash_acts_{i}"]   # (N, T, D)
        d = dn[f"dpo_acts_{i}"]
        # Per-dim std at chunk pos 0, then average across dims
        sv = v[:, 0, :].std(axis=0).mean()
        sd = d[:, 0, :].std(axis=0).mean()
        rows.append((str(t).replace("worlds/l/", "").replace(".json", ""), sv, sd))

    # ---- F8a: diverging bar chart of (σ_ratio − 1) per task ----
    rows.sort(key=lambda r: r[2] / max(r[1], 1e-6))  # sort by σ ratio ascending
    labels = [r[0] for r in rows]
    sv_arr = np.array([r[1] for r in rows])
    sd_arr = np.array([r[2] for r in rows])
    # Positive bars = DEFLECT wider, negative = narrower. Baseline = 0 (σ ratio = 1).
    ratio_arr = sd_arr / np.maximum(sv_arr, 1e-6)
    deviation = ratio_arr - 1.0   # >0 = widened, <0 = narrowed
    order = np.argsort(deviation)  # ascending (most narrowed on top)
    sorted_dev   = deviation[order]
    sorted_labels = np.array(labels)[order]

    fig, ax = plt.subplots(figsize=(7.7, 5.3))
    ys = np.arange(len(sorted_labels))
    bar_colors = [COLORS["DEFLECT"] if d >= 0 else "#888888" for d in sorted_dev]
    ax.barh(ys, sorted_dev, color=bar_colors, edgecolor="black", linewidth=0.4,
            height=0.7)
    ax.axvline(0, color="black", linewidth=1.0)
    # Value labels at bar tips
    for y, d in zip(ys, sorted_dev):
        side = "left" if d < 0 else "right"
        offset = -0.005 if d < 0 else 0.005
        ax.text(d + offset, y, f"{d+1:.2f}", va="center", ha=side,
                fontsize=12, fontweight="bold" if abs(d) > 0.10 else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels(sorted_labels, fontsize=12)
    ax.set_xlabel(r"$\sigma_{\mathrm{DEFLECT}} / \sigma_{\mathrm{VLASH}} - 1$"
                  "  (← narrower | wider →)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # Median annotation
    med = np.median(ratio_arr)
    ax.text(0.97, 0.03,
            f"median ratio = {med:.2f}\n6D mean ratio = {sd_arr.mean()/sv_arr.mean():.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                                  boxstyle="round,pad=0.3"))
    plt.tight_layout()
    plt.savefig(OUT / "F8a_per_task_variance.pdf", bbox_inches="tight")
    plt.close()

    # ---- F8b: scatter VLASH-σ vs DEFLECT-σ with diagonal reference ----
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(sv_arr, sd_arr, s=50, c=COLORS["DEFLECT"],
               edgecolors="black", linewidth=0.6, zorder=5)
    lo = min(sv_arr.min(), sd_arr.min()) * 0.9
    hi = max(sv_arr.max(), sd_arr.max()) * 1.08
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\sigma_{\mathrm{VLASH}}$", fontsize=13)
    ax.set_ylabel(r"$\sigma_{\mathrm{DEFLECT}}$", fontsize=13)
    ax.tick_params(labelsize=8)
    for x, y, lbl in zip(sv_arr, sd_arr, labels):
        ax.annotate(lbl, (x, y), xytext=(4, 3), textcoords="offset points",
                    fontsize=11, alpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "F8b_variance_scatter.pdf", bbox_inches="tight")
    plt.close()

    n_below = int((sd_arr < sv_arr).sum())
    print(f"  F8a_per_task_variance.pdf  ({len(labels)} tasks)")
    print(f"  F8b_variance_scatter.pdf   "
          f"({n_below}/{len(labels)} tasks have σ_DEFLECT < σ_VLASH)")
else:
    print("  F8 skipped: distribution_across_tasks npz not yet available "
          f"({dist_npz})")

# F8c — Delay extrapolation: Models A (DPO d∈{1,2}), B (default d∈{1..4}),
# C (d∈{1..7}) all evaluated at d=0..7. Tests zero-shot generalization.
EXTRAP_RUNS = {
    "A: $d_{\\mathrm{train}}\\in\\{1,2\\}$":    "dpo_vanilla_ep24_cosine_d0_dpomax2_lam0.02_270313",
    "B: $d_{\\mathrm{train}}\\in\\{1\\!-\\!4\\}$ (default)": DPO_MAIN_RUN,
    "C: $d_{\\mathrm{train}}\\in\\{1\\!-\\!7\\}$":  "dpo_vanilla_ep24_cosine_d0_async8_lam0.02_270314",
}
extrap_data, missing = {}, []
for label, run in EXTRAP_RUNS.items():
    rd = dpo_run_delay(run)
    if rd is None:
        missing.append(label)
    else:
        extrap_data[label] = rd[0].values

if not missing:
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    cmap = {"A": "#9c52a0", "B": "#d97706", "C": "#1b9e77"}
    markers = {"A": "v", "B": "^", "C": "s"}
    for label, vals in extrap_data.items():
        key = label[0]
        ax.plot(range(8), vals, marker=markers[key], color=cmap[key],
                linewidth=2, markersize=6, label=label)
    ax.plot(range(8), vlash_d.values, "o:", color=COLORS["VLASH"],
            linewidth=1.5, markersize=4, alpha=0.7, label="VLASH (no DPO)")
    ax.set_xticks(range(8))
    ax.set_xlabel("Inference delay $\\Delta$")
    ax.set_ylabel("Success rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "F8c_delay_extrapolation.pdf", bbox_inches="tight")
    plt.close()
    print(f"  F8c_delay_extrapolation.pdf  "
          f"(A avg={extrap_data[list(extrap_data)[0]].mean():.1f}, "
          f"B avg={extrap_data[list(extrap_data)[1]].mean():.1f}, "
          f"C avg={extrap_data[list(extrap_data)[2]].mean():.1f})")
else:
    print(f"  F8c skipped (missing eval): {missing}")

print(f"\nAll figures written to {OUT}")
