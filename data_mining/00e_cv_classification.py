"""
00e_cv_classification.py — CV-Based Demand Stability Classification

For each SKU, computes mean and standard deviation of daily ORDER FREQUENCY
(fi = number of order lines per day) using the distribution identified in
00d_distribution_fitting.py, following the exact thesis formulas:

  Normal Distribution (Eq. 5-6):
    mu    = (1/N) * sum(fi)                      — mean order frequency
    sigma = sqrt((1/N) * sum((fi - f_bar)^2))    — population std (ddof=0)

  Poisson Distribution (Eq. 7-8):
    mu    = lambda  (estimated as mean(fi) = f_bar)
    sigma = sqrt(lambda)

CV = sigma / mu, then classified:
  CV <= 0.5         → 0  Stable demand
  0.5 < CV <= 1.0   → 1  Volatile demand
  CV > 1.0          → 2  Intermittent demand

Combines with ABC labels from 07_kmeans_clusters.csv to produce the
full SKU classification table (ABC × CV class).

Outputs (data_mining/output/):
  00e_cv_classification.csv   — per-SKU CV, cv_class, abc_kmeans, full_class
  00e_cv_bar.png              — proportion bar chart by CV class
  00e_abc_cv_table.png        — ABC × CV crosstab

Prerequisite: run 00d_distribution_fitting.py and 07_kmeans.py first.
Run: python data_mining/00e_cv_classification.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "output")
ORDERS_PATH  = os.path.join(OUTPUT_DIR, "00_clean_orders.csv")
DIST_PATH    = os.path.join(OUTPUT_DIR, "00d_distribution_labels.csv")
CLUSTER_PATH = os.path.join(OUTPUT_DIR, "07_kmeans_clusters.csv")

CV_THRESHOLDS = {0: (0.0, 0.5), 1: (0.5, 1.0), 2: (1.0, float("inf"))}
CV_LABELS     = {0: "Stable", 1: "Volatile", 2: "Intermittent"}
CV_DESC       = {0: "stable", 1: "volatile", 2: "intermittent"}

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — LOAD DATA")
print("=" * 65)

orders = pd.read_csv(ORDERS_PATH)
orders["order_date"] = pd.to_datetime(orders["order_date"], format="%m/%d/%Y %H:%M")
orders["date"]       = orders["order_date"].dt.date

all_days  = sorted(orders["date"].unique())
n_days    = len(all_days)   # N in the thesis formulas
day_index = {d: i for i, d in enumerate(all_days)}

dist_df    = pd.read_csv(DIST_PATH)
cluster_df = pd.read_csv(CLUSTER_PATH)[["item_code", "abc_kmeans"]]

print(f"  Orders         : {len(orders):,}")
print(f"  Days (N)       : {n_days}  ({all_days[0]} → {all_days[-1]})")
print(f"  SKUs with dist : {len(dist_df):,}")
print(f"  SKUs with ABC  : {len(cluster_df):,}")

# ── 2. Build daily ORDER FREQUENCY (fi) per SKU ───────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — BUILD DAILY ORDER FREQUENCY (fi)")
print("=" * 65)

# fi = number of order lines per SKU per day (not quantity — see thesis Eq. 5)
daily = (
    orders.groupby(["item_code", "date"])["order_id"]
    .count()
    .reset_index()
    .rename(columns={"order_id": "daily_orders"})
)

# Pivot to full N-day matrix, filling 0 on days with no orders
pivot = (
    daily.pivot(index="item_code", columns="date", values="daily_orders")
    .reindex(columns=all_days, fill_value=0)
    .fillna(0)
)
print(f"  Daily order-frequency matrix : {pivot.shape[0]:,} SKUs x {pivot.shape[1]} days")

# ── 3. Compute mu, sigma, CV per SKU ─────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — COMPUTE mu, sigma, CV PER SKU")
print("=" * 65)

dist_map = dist_df.set_index("item_code")["best_fit"].to_dict()

results = []
for sku in dist_df["item_code"]:
    if sku in pivot.index:
        fi = pivot.loc[sku].values.astype(float)
    else:
        fi = np.zeros(n_days)

    dist  = dist_map.get(sku, "poisson")
    f_bar = fi.mean()           # (1/N)*sum(fi) — Eq. 5

    if f_bar <= 0:
        mu, sigma, cv = 0.0, 0.0, 0.0
    elif dist == "normal":
        # Eq. 5: mu = (1/N)*sum(fi)
        mu    = f_bar
        # Eq. 6: sigma = sqrt((1/N)*sum((fi - f_bar)^2))  — population std
        sigma = fi.std(ddof=0)
        cv    = sigma / mu if mu > 0 else 0.0
    else:
        # Poisson — Eq. 7-8: mu = lambda, sigma = sqrt(lambda)
        mu    = f_bar           # lambda estimated as mean(fi)
        sigma = np.sqrt(mu)
        cv    = sigma / mu      # = 1/sqrt(lambda)

    # Classify CV
    if cv <= 0.5:
        cv_class = 0
    elif cv <= 1.0:
        cv_class = 1
    else:
        cv_class = 2

    results.append({
        "item_code":          sku,
        "distribution":       dist,
        "mean_order_freq":    round(mu,    4),
        "std_order_freq":     round(sigma, 4),
        "cv":                 round(cv,    4),
        "cv_class":           cv_class,
    })

cv_df = pd.DataFrame(results)
print(f"  CV computed for {len(cv_df):,} SKUs")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — CV CLASS SUMMARY")
print("=" * 65)

print(f"\n  {'CV Class':<5}  {'Label':<15}  {'CV Range':<20}  {'SKUs':>8}  {'%':>7}")
print(f"  {'-'*60}")
for cls in [0, 1, 2]:
    cnt = (cv_df["cv_class"] == cls).sum()
    pct = cnt / len(cv_df) * 100
    lo, hi = CV_THRESHOLDS[cls]
    hi_str = f"{hi:.1f}" if hi != float("inf") else "inf"
    print(f"  {cls:<5}  {CV_LABELS[cls]:<15}  {lo:.1f} < CV <= {hi_str:<8}  {cnt:>8,}  {pct:>6.1f}%")

# ── 5. Merge with ABC labels ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — MERGE WITH ABC LABELS")
print("=" * 65)

cv_df = cv_df.merge(cluster_df, on="item_code", how="left")
cv_df["abc_kmeans"] = cv_df["abc_kmeans"].fillna("C")

abc_demand = {"A": "high demand", "B": "medium demand", "C": "low demand"}
cv_df["full_class"] = (
    cv_df["abc_kmeans"].map(abc_demand) + " - " + cv_df["cv_class"].map(CV_DESC)
)

# ── 6. ABC × CV crosstab ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — ABC x CV CLASSIFICATION TABLE")
print("=" * 65)

print(f"\n  {'Cluster':<9} {'CV Class':<10} {'Description':<34} {'SKU Count':>10}")
print(f"  {'-'*67}")
for abc in ["A", "B", "C"]:
    for cv_cls in [0, 1, 2]:
        sub = cv_df[(cv_df["abc_kmeans"] == abc) & (cv_df["cv_class"] == cv_cls)]
        cnt = len(sub)
        desc = f"{abc_demand[abc]} - {CV_DESC[cv_cls]}"
        cnt_str = f"{cnt:,}" if cnt > 0 else "-"
        print(f"  {abc:<9} {cv_cls:<10} {desc:<34} {cnt_str:>10}")

# ── 7. Bar chart — proportion by CV class ─────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — PLOTS")
print("=" * 65)

cv_counts  = cv_df["cv_class"].value_counts().sort_index()
cv_labels  = [CV_LABELS[c] + " Demand" for c in cv_counts.index]
cv_pcts    = cv_counts / len(cv_df) * 100
bar_colors = ["darkorange", "steelblue", "seagreen"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(cv_labels, cv_pcts.values, color=bar_colors, edgecolor="white", width=0.5)
for bar, pct in zip(bars, cv_pcts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=11)

# Add numeric class labels below x-axis labels
ax.set_xticks(range(len(cv_labels)))
ax.set_xticklabels([f"{lbl}\n{cls}" for lbl, cls in zip(cv_labels, cv_counts.index)])

ax.set_title("SKU Classiffication", fontsize=13)
ax.set_ylabel("Proportion of SKUs (%)")
ax.set_ylim(0, max(cv_pcts.values) * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
bar_png = os.path.join(OUTPUT_DIR, "00e_cv_bar.png")
plt.savefig(bar_png, dpi=150)
plt.close()
print(f"  Saved: output/00e_cv_bar.png")

# ── 8. Save ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 — SAVE OUTPUTS")
print("=" * 65)

name_map = dist_df[["item_code", "item_name"]].drop_duplicates()
out_df   = cv_df.merge(name_map, on="item_code", how="left")
out_cols = [
    "item_code", "item_name", "abc_kmeans",
    "distribution", "mean_order_freq", "std_order_freq", "cv", "cv_class", "full_class",
]
out_df[out_cols].to_csv(os.path.join(OUTPUT_DIR, "00e_cv_classification.csv"), index=False)
print(f"  Saved: output/00e_cv_classification.csv  ({len(out_df):,} rows)")

print(f"\nDone — CV classification complete.")
print(f"  Stable={(cv_df['cv_class']==0).sum():,}  "
      f"Volatile={(cv_df['cv_class']==1).sum():,}  "
      f"Intermittent={(cv_df['cv_class']==2).sum():,}")
