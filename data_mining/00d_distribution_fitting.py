"""
00d_distribution_fitting.py — Per-SKU Demand Distribution Fitting

For each SKU, builds a daily ORDER FREQUENCY time series over the N-day window
(fi = number of distinct orders placed per day, zero on days with no orders),
then fits Poisson and Normal distributions using a chi-square goodness-of-fit
test. The best-fitting distribution (highest chi-square p-value) is assigned.

Method:
  - fi          : order frequency for day i (count of order lines per day, 0 if none)
  - f_bar (mu)  : mean order frequency = (1/N) * sum(fi)
  - Poisson GOF : lambda = f_bar; chi-square GOF on full series
  - Normal GOF  : mu = f_bar, sigma = sqrt((1/N)*sum((fi-f_bar)^2)); chi-square on full series
  - Best fit    : distribution with highest chi-square p-value (least rejected)
  - Sparse SKUs (< MIN_DAYS demand days) default to Poisson

Outputs (data_mining/output/):
  00d_distribution_labels.csv   — per-SKU best-fit distribution + test stats
  00d_distribution_bar.png      — bar chart of SKU counts per distribution type

Prerequisite: run 00_data_cleaning.py then 00b_data_filtering.py first.
Run: python data_mining/00d_distribution_fitting.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

warnings.filterwarnings("ignore")

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output")
ORDERS_PATH   = os.path.join(OUTPUT_DIR, "00_clean_orders.csv")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "00b_sku_demand_features.csv")

MIN_DAYS = 3   # minimum distinct demand days to run GOF test reliably

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — LOAD CLEAN ORDERS")
print("=" * 65)

orders = pd.read_csv(ORDERS_PATH)
orders["order_date"] = pd.to_datetime(orders["order_date"], format="%m/%d/%Y %H:%M")
orders["date"]       = orders["order_date"].dt.date

all_days  = sorted(orders["date"].unique())
n_days    = len(all_days)
day_index = {d: i for i, d in enumerate(all_days)}

print(f"  Orders loaded : {len(orders):,}")
print(f"  Date range    : {all_days[0]} → {all_days[-1]}  ({n_days} days)")

skus_df  = pd.read_csv(FEATURES_PATH)
sku_list = skus_df["item_code"].tolist()
print(f"  SKUs to test  : {len(sku_list):,}")

# ── 2. Build daily ORDER COUNT (fi) per SKU ───────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — BUILD DAILY ORDER FREQUENCY (fi) PER SKU")
print("=" * 65)

# fi = number of order lines per SKU per day (order frequency, not quantity)
daily = (
    orders.groupby(["item_code", "date"])["order_id"]
    .count()
    .reset_index()
    .rename(columns={"order_id": "daily_orders"})
)
print(f"  Daily order-count rows : {len(daily):,}")

# ── 3. GOF test per SKU ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — GOODNESS-OF-FIT TEST PER SKU")
print("=" * 65)


def poisson_gof(fi_series, lam):
    """
    Chi-square GOF for Poisson on the full daily order-frequency series.
    lam = mean(fi) = f_bar  (Eq. 7: mu = lambda)
    """
    if lam <= 0:
        return 0.0, 1.0
    n = len(fi_series)
    max_val = max(int(fi_series.max()) + 1, int(lam * 3) + 1)
    bins    = list(range(0, max_val + 1))
    observed, _ = np.histogram(fi_series, bins=bins + [max_val + 1])
    expected    = np.array([stats.poisson.pmf(k, lam) * n for k in bins])
    # Merge bins until expected >= 5
    obs_m, exp_m = [], []
    oa, ea = 0, 0
    for o, e in zip(observed, expected):
        oa += o; ea += e
        if ea >= 5:
            obs_m.append(oa); exp_m.append(ea)
            oa, ea = 0, 0
    if ea > 0:
        if obs_m:
            obs_m[-1] += oa; exp_m[-1] += ea
        else:
            obs_m.append(oa); exp_m.append(ea)
    obs_m = np.array(obs_m, dtype=float)
    exp_m = np.array(exp_m, dtype=float)
    if len(obs_m) < 2:
        return 0.0, 1.0
    stat = np.sum((obs_m - exp_m) ** 2 / exp_m)
    df   = max(len(obs_m) - 1 - 1, 1)   # bins - 1 - 1 estimated param (lambda)
    return stat, 1 - chi2.cdf(stat, df)


def normal_gof(fi_series, n_bins=10):
    """
    Chi-square GOF for Normal on the full daily order-frequency series.
    mu    = (1/N) * sum(fi)                    — Eq. 5
    sigma = sqrt((1/N) * sum((fi - f_bar)^2))  — Eq. 6  (population std, ddof=0)
    """
    n     = len(fi_series)
    mu    = fi_series.mean()                    # (1/N)*sum(fi)
    sigma = fi_series.std(ddof=0)               # population std (Eq. 6)
    if sigma <= 0:
        return 0.0, 0.0
    dist        = stats.norm(mu, sigma)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges   = np.unique(np.percentile(fi_series, percentiles))
    if len(bin_edges) < 3:
        return 0.0, 0.0
    observed, _ = np.histogram(fi_series, bins=bin_edges)
    expected    = np.diff(dist.cdf(bin_edges)) * n
    mask = expected >= 5
    if mask.sum() < 2:
        return 0.0, 0.0
    observed = observed[mask].astype(float)
    expected = expected[mask]
    stat = np.sum((observed - expected) ** 2 / expected)
    df   = max(mask.sum() - 1 - 2, 1)          # bins - 1 - 2 estimated params (mu, sigma)
    return stat, 1 - chi2.cdf(stat, df)


results = []

print(f"  Testing {len(sku_list):,} SKUs...")
for i, sku in enumerate(sku_list):
    sku_daily = daily[daily["item_code"] == sku]

    # Build full N-day fi series (0 on days with no orders)
    fi_full = np.zeros(n_days)
    for _, row in sku_daily.iterrows():
        idx = day_index.get(row["date"])
        if idx is not None:
            fi_full[idx] = row["daily_orders"]

    days_with_demand = (fi_full > 0).sum()
    f_bar            = fi_full.mean()           # (1/N)*sum(fi) — Eq. 5

    if days_with_demand < MIN_DAYS or f_bar <= 0:
        results.append({
            "item_code":     sku,
            "best_fit":      "poisson",
            "poisson_p":     np.nan,
            "normal_p":      np.nan,
            "note":          "sparse",
        })
        continue

    # Poisson GOF (lambda = f_bar)
    _, p_pois = poisson_gof(fi_full, f_bar)

    # Normal GOF (mu = f_bar, sigma = population std)
    _, p_norm = normal_gof(fi_full)

    best = "poisson" if p_pois >= p_norm else "normal"

    results.append({
        "item_code":     sku,
        "best_fit":      best,
        "poisson_p":     round(p_pois, 4),
        "normal_p":      round(p_norm, 4),
        "note":          "",
    })

    if (i + 1) % 1000 == 0:
        print(f"    {i+1:,} / {len(sku_list):,} done...")

print(f"  Done.")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — DISTRIBUTION SUMMARY")
print("=" * 65)

results_df  = pd.DataFrame(results)
dist_counts = results_df["best_fit"].value_counts()

print(f"\n  {'Distribution':<15}  {'SKUs':>8}  {'%':>7}")
print(f"  {'-'*34}")
for d, cnt in dist_counts.items():
    pct = cnt / len(results_df) * 100
    print(f"  {d:<15}  {cnt:>8,}  {pct:>6.1f}%")
print(f"  {'Total':<15}  {len(results_df):>8,}  100.0%")

sparse = (results_df["note"] == "sparse").sum()
print(f"\n  Sparse SKUs (< {MIN_DAYS} demand days, defaulted to Poisson): {sparse:,}")

# ── 5. Bar chart ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — BAR CHART")
print("=" * 65)

plot_dists  = [d for d in ["poisson", "normal"] if d in dist_counts.index]
plot_counts = [dist_counts[d] for d in plot_dists]
plot_labels = [d.capitalize() for d in plot_dists]
plot_colors = ["steelblue", "tomato"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(plot_labels, plot_counts, color=plot_colors, edgecolor="white", width=0.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=11)
ax.set_title("Goodness of Fit Test", fontsize=13)
ax.set_xlabel("Distribution Type")
ax.set_ylabel("Number of SKUs")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
bar_png = os.path.join(OUTPUT_DIR, "00d_distribution_bar.png")
plt.savefig(bar_png, dpi=150)
plt.close()
print(f"  Saved: output/00d_distribution_bar.png")

# ── 6. Save ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — SAVE OUTPUTS")
print("=" * 65)

out_df = skus_df[["item_code", "item_name", "order_frequency",
                   "total_qty", "avg_qty_per_order", "days_ordered"]].merge(
    results_df, on="item_code", how="left"
)
out_df.to_csv(os.path.join(OUTPUT_DIR, "00d_distribution_labels.csv"), index=False)
print(f"  Saved: output/00d_distribution_labels.csv  ({len(out_df):,} rows)")

print(f"\nDone — Distribution fitting complete.")
print(f"  {len(results_df):,} SKUs tested  →  "
      + "  ".join([f"{d.capitalize()}={dist_counts.get(d, 0):,}"
                   for d in ["poisson", "normal"]]))
