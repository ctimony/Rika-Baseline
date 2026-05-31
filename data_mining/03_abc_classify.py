"""
03_abc_classify.py
==================
ABC classification of SKUs using K-Means clustering (k=3)
based on demand characteristics from clean_orders.

Features used:
- order_frequency : number of unique orders containing the SKU
- total_quantity  : total units ordered across all orders

Steps:
1. Compute demand features from 02_clean_orders.csv
2. Normalize features (StandardScaler)
3. Elbow method (k=1..10) to confirm optimal k
4. K-Means k=3, assign labels A/B/C (A=highest demand)
5. Save results and elbow plot

Output:
  data_mining/output/03_abc_skus.csv
  data_mining/output/03_elbow_plot.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT        = os.path.dirname(os.path.abspath(__file__))
ORDERS_PATH = os.path.join(ROOT, "output", "02_clean_orders.csv")
SKUS_PATH   = os.path.join(ROOT, "output", "01_clean_skus_pod3.csv")
OUTPUT_PATH = os.path.join(ROOT, "output", "03_abc_skus.csv")
ELBOW_PATH  = os.path.join(ROOT, "output", "03_elbow_plot.png")

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("  ABC Classification — K-Means Clustering (k=3)")
    print("=" * 60)

    orders = pd.read_csv(ORDERS_PATH)
    skus   = pd.read_csv(SKUS_PATH)
    orders["item_code"] = orders["item_code"].astype(str)
    skus["item_code"]   = skus["item_code"].astype(str)

    # ── Step 1: Compute demand features per SKU ────────────────
    demand = orders.groupby("item_code").agg(
        order_frequency=("order_id", "nunique"),
        total_quantity=("item_quantity", "sum"),
    ).reset_index()

    # SKUs in clean_skus but with 0 orders get frequency=0
    demand = skus[["item_code"]].merge(demand, on="item_code", how="left").fillna(0)
    print(f"\n  SKUs with demand data : {(demand['order_frequency'] > 0).sum():,}")
    print(f"  SKUs with zero orders : {(demand['order_frequency'] == 0).sum():,}")

    # ── Step 2: Log transform + Normalize ─────────────────────
    demand["log_freq"] = np.log1p(demand["order_frequency"])
    features = demand[["log_freq"]].values
    scaler   = StandardScaler()
    X        = scaler.fit_transform(features)

    # ── Step 3: Elbow method ───────────────────────────────────
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)

    print("\n  WCSS per k:")
    for k, w in enumerate(wcss, 1):
        print(f"    k={k:2d}  WCSS={w:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, "bo-", linewidth=2, markersize=6)
    plt.title("Elbow Method: WCSS vs. Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.xticks(range(1, 11))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=150)
    plt.close()
    print(f"\n  Elbow plot saved → {ELBOW_PATH}")

    # ── Step 4: K-Means k=3 ────────────────────────────────────
    km3 = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    demand["cluster"] = km3.fit_predict(X)

    # Rank clusters by mean order_frequency: highest = A
    cluster_rank = (
        demand.groupby("cluster")["order_frequency"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    cluster_rank["abc_class"] = ["A", "B", "C"]
    demand = demand.merge(cluster_rank[["cluster", "abc_class"]], on="cluster")

    # ── Step 5: Summary ────────────────────────────────────────
    print("\n  Cluster composition:")
    total = len(demand)
    for cls in ["A", "B", "C"]:
        n = (demand["abc_class"] == cls).sum()
        print(f"    Class {cls}: {n:,} SKUs ({n/total*100:.2f}%)")

    print(f"\n  Total SKUs classified : {total:,}")

    # ── Save ───────────────────────────────────────────────────
    result = demand[["item_code", "order_frequency", "total_quantity", "abc_class"]]
    result = result.sort_values("order_frequency", ascending=False).reset_index(drop=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved → {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
