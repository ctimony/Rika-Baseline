"""
07_kmeans.py — K-Means Clustering with Elbow Method

Reads from output/00b_sku_demand_features.csv (produced by 00b_data_filtering.py).
Oversized SKUs are already excluded.

Feature:
  order_frequency   — how often the SKU is ordered (log1p + StandardScaler applied)

Optimal k:
  Determined by the Elbow Method (WCSS) — largest deceleration in WCSS drop
  (2nd difference of consecutive WCSS drops).

ABC labelling (post-clustering):
  Clusters are ranked by centroid order_frequency (descending):
    Rank 0       (highest demand) → A
    Rank 1..k-2  (middle)        → B  (all middle clusters merged)
    Rank k-1     (lowest demand) → C

Log-transform rationale:
  Demand data is strongly right-skewed (skew ~8). Log1p corrects this so
  K-Means measures proportional differences rather than absolute ones.

Run: python data_mining/07_kmeans.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "00b_sku_demand_features.csv")

FEATURES = ["order_frequency"]
K_MIN    = 1
K_MAX    = 10

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — LOAD SKU DEMAND FEATURES")
print("=" * 65)

df = pd.read_csv(FEATURES_PATH)
print(f"  SKUs loaded : {len(df):,}")
print(f"\n  Raw feature statistics:")
for feat in FEATURES:
    print(f"  {feat:<22} skew={df[feat].skew():>6.2f}  "
          f"min={df[feat].min():.1f}  "
          f"max={df[feat].max():.1f}  "
          f"mean={df[feat].mean():.1f}")

# ── 2. Log-transform then scale ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — LOG-TRANSFORM & STANDARDISE")
print("=" * 65)

X_log    = np.log1p(df[FEATURES].values)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

log_df = pd.DataFrame(X_log, columns=FEATURES)
print(f"\n  Skewness before and after log1p:")
for feat in FEATURES:
    print(f"  {feat:<22} before={df[feat].skew():>6.2f}  ->  after={log_df[feat].skew():>6.2f}")

# ── 3. WCSS — Elbow Method ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 3 — WCSS ELBOW METHOD  (k = {K_MIN} to {K_MAX})")
print("=" * 65)

k_range = list(range(K_MIN, K_MAX + 1))
wcss    = []

print(f"\n  {'k':>3}  {'WCSS':>10}")
print(f"  {'-'*16}")
for k in k_range:
    km_fit = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_fit.fit(X_scaled)
    wcss.append(km_fit.inertia_)
    print(f"  {k:>3}  {wcss[-1]:>10,.2f}")

# Detect elbow — largest 2nd difference in WCSS drops
drops       = [wcss[i-1] - wcss[i] for i in range(1, len(wcss))]
delta_drops = [drops[i-1] - drops[i] for i in range(1, len(drops))]

elbow_idx = int(np.argmax(delta_drops))
elbow_k   = k_range[elbow_idx + 2]
print(f"\n  Elbow detected at k = {elbow_k}  (largest WCSS drop deceleration)")

# ── 4. Optimal k ──────────────────────────────────────────────────────────────
optimal_k = elbow_k
print(f"\n  Optimal k = {optimal_k}")

# ── 5. Elbow plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_range, wcss, marker="o", color="steelblue", linewidth=2, markersize=7)
for k, w in zip(k_range, wcss):
    ax.annotate(f"{w:,.2f}", xy=(k, w),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=7, color="dimgray")
ax.set_title("Elbow Method: WCSS vs. Number of Clusters", fontsize=13)
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
ax.set_xticks(k_range)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_kmeans_elbow.png"), dpi=150)
plt.close()
print(f"\n  Saved: output/07_kmeans_elbow.png")

# Save WCSS table
wcss_df = pd.DataFrame({"k": k_range, "wcss": [round(w, 2) for w in wcss]})
wcss_df.to_csv(os.path.join(OUTPUT_DIR, "07_wcss_table.csv"), index=False)
print(f"  Saved: output/07_wcss_table.csv")

# ── 6. Final K-Means at optimal k ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 4 — FINAL K-MEANS  (k = {optimal_k})")
print("=" * 65)

km_final      = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df["cluster"] = km_final.fit_predict(X_scaled)

# Inverse-transform centroids to original scale
centroid_log  = scaler.inverse_transform(km_final.cluster_centers_)
centroid_orig = np.expm1(centroid_log)
centroids_df  = pd.DataFrame(centroid_orig, columns=FEATURES)
centroids_df.insert(0, "cluster", list(range(optimal_k)))
centroids_df["sku_count"] = [(df["cluster"] == k).sum() for k in range(optimal_k)]

# Rank clusters by order_frequency descending
centroids_df = centroids_df.sort_values(
    "order_frequency", ascending=False
).reset_index(drop=True)
centroids_df["rank"] = centroids_df.index   # 0 = highest demand

print(f"\n  Clusters ranked by centroid order_frequency (original scale):")
print(f"  {'Rank':<6} {'Cluster':<9} {'Freq':>8} {'SKUs':>7}")
print(f"  {'-'*34}")
for _, row in centroids_df.iterrows():
    print(f"  {int(row['rank']):<6} {int(row['cluster']):<9}"
          f" {row['order_frequency']:>8.1f}"
          f" {int(row['sku_count']):>7,}")

# ── 7. ABC labelling ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — ABC LABEL ASSIGNMENT")
print("=" * 65)

def abc_label(rank, total):
    if rank == 0:          return "A"
    if rank == total - 1:  return "C"
    return "B"

centroids_df["abc_kmeans"] = [abc_label(i, optimal_k) for i in range(optimal_k)]
cluster_to_abc             = centroids_df.set_index("cluster")["abc_kmeans"].to_dict()
df["abc_kmeans"]           = df["cluster"].map(cluster_to_abc)

print(f"\n  {'Cluster':<9} {'Rank':<6} {'ABC':<5} {'SKUs':>7} {'%':>7} {'Freq (mean)':>12}")
print(f"  {'-'*50}")
for _, row in centroids_df.iterrows():
    pct = int(row["sku_count"]) / len(df) * 100
    print(f"  {int(row['cluster']):<9} {int(row['rank']):<6} {row['abc_kmeans']:<5}"
          f" {int(row['sku_count']):>7,} {pct:>6.2f}%"
          f" {row['order_frequency']:>12.1f}")

print(f"\n  ABC Summary:")
for cls in ["A", "B", "C"]:
    sub = df[df["abc_kmeans"] == cls]
    if len(sub) == 0:
        continue
    pct = len(sub) / len(df) * 100
    print(f"    {cls}: {len(sub):>5,} SKUs ({pct:.2f}%)"
          f"  |  freq mean={sub['order_frequency'].mean():.1f}"
          f"  min={sub['order_frequency'].min():.0f}"
          f"  max={sub['order_frequency'].max():.0f}")

# ── 8. Bar chart ───────────────────────────────────────────────────────────────
abc_counts = df["abc_kmeans"].value_counts().reindex(["A", "B", "C"])
colors     = {"A": "steelblue", "B": "seagreen", "C": "sandybrown"}

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["A", "B", "C"],
              abc_counts.values,
              color=[colors[c] for c in ["A", "B", "C"]],
              edgecolor="white", width=0.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=11)
ax.set_title("SKU Cluster by K-Means", fontsize=13)
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of SKUs")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_sku_cluster_bar.png"), dpi=150)
plt.close()
print(f"\n  Saved: output/07_sku_cluster_bar.png")

# ── 9. Save ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — SAVE OUTPUTS")
print("=" * 65)

out_cols = [
    "item_code", "item_name",
    "order_frequency", "total_qty", "avg_qty_per_order",
    "std_qty", "demand_cv", "days_ordered", "avg_basket_size",
    "box_length", "box_width", "box_height", "box_volume", "item_volume",
    "cluster", "abc_kmeans",
]
df[out_cols].to_csv(os.path.join(OUTPUT_DIR, "07_kmeans_clusters.csv"), index=False)
print(f"  Saved: output/07_kmeans_clusters.csv  ({len(df):,} rows)")

centroids_df.to_csv(os.path.join(OUTPUT_DIR, "07_kmeans_centroids.csv"), index=False)
print(f"  Saved: output/07_kmeans_centroids.csv")

print(f"\nDone — K-Means complete.")
print(f"  Optimal k = {optimal_k}  (Elbow Method)")
print(f"  {len(df):,} SKUs clustered  ->  ABC: "
      + "  ".join([f"{c}={len(df[df['abc_kmeans']==c]):,}" for c in ['A','B','C']]))
