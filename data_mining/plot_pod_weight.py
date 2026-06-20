"""
plot_pod_weight.py
==================
Regenerate result/pod_weight_distribution.png from the current pods.csv.

Pod weight = Σ (total_item_weight) over the filled slots of a pod, i.e. the
summed mass of the products it currently stores (matches Pod.mass / Eq m_p).
Run after any change to pods.csv (sampling, slot scheme, SKU set).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT     = os.path.dirname(os.path.abspath(__file__))
PODS_CSV = os.path.join(ROOT, "..", "pods.csv")
OUT_PNG  = os.path.join(ROOT, "..", "result", "pod_weight_distribution.png")


def main():
    pods = pd.read_csv(PODS_CSV)
    filled = pods[pods["item"] != -1]

    # Mass per pod = sum of total_item_weight across its filled slots.
    pod_weight = filled.groupby("pod_id")["total_item_weight"].sum()
    n_pods = pod_weight.shape[0]
    n_sku  = filled["item"].nunique()

    mean_w   = pod_weight.mean()
    median_w = pod_weight.median()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(pod_weight, bins=40, color="#6ba3d6", edgecolor="black", linewidth=0.6)
    ax.axvline(mean_w,   color="red",   linestyle="--", linewidth=2,
               label=f"Mean = {mean_w:.1f} kg")
    ax.axvline(median_w, color="green", linestyle=":",  linewidth=2,
               label=f"Median = {median_w:.1f} kg")
    ax.set_title(f"Distribution of Pod Weights (Pod3, {n_sku:,} SKUs)", fontsize=15)
    ax.set_xlabel("Pod Weight (kg)", fontsize=12)
    ax.set_ylabel("Number of Pods", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"  Pods            : {n_pods:,}")
    print(f"  SKUs (unique)   : {n_sku:,}")
    print(f"  Mean pod weight : {mean_w:.1f} kg")
    print(f"  Median          : {median_w:.1f} kg")
    print(f"  Min / Max       : {pod_weight.min():.1f} / {pod_weight.max():.1f} kg")
    print(f"  Total mass      : {pod_weight.sum():,.0f} kg")
    print(f"  Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
