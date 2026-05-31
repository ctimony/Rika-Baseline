"""
06_sampling.py
==============
Class-proportion weighted sampling of SKUs for simulation.

Method (based on thesis approach):
1. Compute importance score per SKU as geometric mean of 3 factors:
   - composite  = mean_daily_demand * order_frequency
   - stability  = 1 / ADI
   - variability = 1 + CV²  (1 added to avoid zero for smooth SKUs)
   importance = (composite * stability * variability)^(1/3)

2. Normalize importance score within each ABC class → sampling probability

3. Weighted sampling per class, proportional to original class distribution
   Target: 3,000 SKUs total

Output: data_mining/output/06_sampled_skus.csv
"""

import os
import numpy as np
import pandas as pd

ROOT        = os.path.dirname(os.path.abspath(__file__))
DICT_PATH   = os.path.join(ROOT, "..", "items_dictionary.csv")
OUTPUT_PATH = os.path.join(ROOT, "output", "06_sampled_skus.csv")

TARGET_SKU  = 3000
RANDOM_SEED = 42


def main():
    print("=" * 60)
    print("  SKU Sampling — Class-Proportion Weighted")
    print("=" * 60)

    df = pd.read_csv(DICT_PATH)
    df["item_code"] = df["item_code"].astype(str)

    total = len(df)
    print(f"\n  Input SKUs          : {total:,}")
    print(f"  Target sample       : {TARGET_SKU:,}")

    # ── Importance score ───────────────────────────────────────
    df["composite"]   = df["mean_daily_demand"] * df["item_order_frequency"]
    df["stability"]   = 1.0 / df["adi"].clip(lower=0.01)
    df["variability"] = 1.0 + df["cv2"]

    # Geometric mean of 3 factors
    df["importance"] = (
        df["composite"] * df["stability"] * df["variability"]
    ).clip(lower=0).pow(1/3)

    # ── Proportional target per ABC class ──────────────────────
    class_counts = df["item_class"].value_counts().sort_index()
    class_targets = {}
    allocated = 0
    classes = sorted(class_counts.index)
    for i, cls in enumerate(classes):
        if i == len(classes) - 1:
            # Last class gets remainder to ensure exact total
            class_targets[cls] = TARGET_SKU - allocated
        else:
            n = round(class_counts[cls] / total * TARGET_SKU)
            class_targets[cls] = n
            allocated += n

    print(f"\n  Sampling targets per class:")
    for cls in classes:
        n_pop = class_counts[cls]
        n_samp = class_targets[cls]
        print(f"    {cls}: {n_samp:,} from {n_pop:,} ({n_samp/n_pop*100:.1f}% of class)")

    # ── Weighted sampling per class ────────────────────────────
    sampled_parts = []
    np.random.seed(RANDOM_SEED)

    for cls in classes:
        sub = df[df["item_class"] == cls].copy()
        n_samp = class_targets[cls]

        # Normalize importance → probability
        total_imp = sub["importance"].sum()
        if total_imp > 0:
            sub["prob"] = sub["importance"] / total_imp
        else:
            sub["prob"] = 1.0 / len(sub)

        # Cap sample to available SKUs
        n_samp = min(n_samp, len(sub))
        sampled = sub.sample(n=n_samp, weights="prob", random_state=RANDOM_SEED)
        sampled_parts.append(sampled)

    result = pd.concat(sampled_parts).reset_index(drop=True)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n  Final sample        : {len(result):,} SKUs")
    print(f"\n  ABC distribution:")
    for cls in classes:
        n = (result["item_class"] == cls).sum()
        print(f"    {cls}: {n:,} ({n/len(result)*100:.1f}%)")

    print(f"\n  Demand pattern:")
    for pat in ["smooth", "erratic", "intermittent", "lumpy"]:
        n = (result["demand_pattern"] == pat).sum()
        print(f"    {pat:13s}: {n:,} ({n/len(result)*100:.1f}%)")

    print(f"\n  Slots needed        : {result['slots_needed'].sum():,}")
    pods_needed = int(np.ceil(result["slots_needed"].sum() / 20))
    print(f"  Pods needed (20/pod): {pods_needed:,}")

    # Save — keep all columns from items_dictionary
    drop_cols = ["composite", "stability", "variability", "importance", "prob"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])
    result = result.sort_values("item_order_frequency", ascending=False).reset_index(drop=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
