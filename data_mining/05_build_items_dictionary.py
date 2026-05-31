"""
05_build_items_dictionary.py
============================
Rebuilds items_dictionary.csv from clean pipeline outputs.

Formula (Minimum Inventory Level — Equation 10):
  initial_qty = mean_daily_demand * cov_days
                + Z * std_daily_demand * sqrt(cov_days)

Coverage days per demand pattern:
  smooth / erratic  : 1 day
  intermittent      : 3 days
  lumpy             : 4 days

Z score per ABC × CV class:
  A × CV0 : 2.33 (99%)    B × CV0 : 1.64 (95%)    C × CV0 : 1.28 (90%)
  A × CV1 : 1.64 (95%)    B × CV1 : 1.28 (90%)    C × CV1 : 1.04 (85%)
  A × CV2 : 1.28 (90%)    B × CV2 : 1.04 (85%)    C × CV2 : 0.84 (80%)

ROP (lead time = 1/8 day = 1 shift):
  rop = mean_daily_demand * (1/8) + Z * std_daily_demand * sqrt(1/8)
  minimum 1

Inputs:
  data_mining/output/01_clean_skus_pod3.csv
  data_mining/output/04_cv_classification.csv

Output: items_dictionary.csv  (project root)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT      = os.path.dirname(os.path.abspath(__file__))
SKUS_PATH = os.path.join(ROOT, "output", "01_clean_skus_pod3.csv")
CV_PATH   = os.path.join(ROOT, "output", "04_cv_classification.csv")
OUT_PATH  = os.path.join(ROOT, "..", "items_dictionary.csv")

POD3_SLOT_VOL = 60_000  # cm³

COVERAGE_DAYS = {
    ("A", "smooth"):       1,
    ("A", "erratic"):      1,
    ("A", "intermittent"): 2,
    ("A", "lumpy"):        3,
    ("B", "smooth"):       2,
    ("B", "erratic"):      2,
    ("B", "intermittent"): 3,
    ("B", "lumpy"):        4,
    ("C", "smooth"):       3,
    ("C", "erratic"):      3,
    ("C", "intermittent"): 4,
    ("C", "lumpy"):        4,
}

Z_MAP = {
    ("A", 0): norm.ppf(0.99),  # 2.3263
    ("A", 1): norm.ppf(0.95),  # 1.6449
    ("A", 2): norm.ppf(0.90),  # 1.2816
    ("B", 0): norm.ppf(0.95),  # 1.6449
    ("B", 1): norm.ppf(0.95),  # 1.6449
    ("B", 2): norm.ppf(0.90),  # 1.2816
    ("C", 0): norm.ppf(0.90),  # 1.2816
    ("C", 1): norm.ppf(0.90),  # 1.2816
    ("C", 2): norm.ppf(0.90),  # 1.2816
}

LEAD_TIME_ROP = 1 / 8  # 1 shift


def main():
    print("=" * 60)
    print("  Build items_dictionary.csv")
    print("=" * 60)

    skus = pd.read_csv(SKUS_PATH)
    cv   = pd.read_csv(CV_PATH)

    skus["item_code"] = skus["item_code"].astype(str)
    cv["item_code"]   = cv["item_code"].astype(str)

    print(f"\n  Clean SKUs          : {len(skus):,}")
    print(f"  CV classification   : {len(cv):,}")

    # ── Drop columns that will be replaced by cv_classification ─
    drop_cols = [c for c in ["mean_daily_demand", "std_daily_demand", "cv",
                             "item_class", "abc_xyz", "rop_global", "slots_needed"]
                 if c in skus.columns]
    skus = skus.drop(columns=drop_cols)

    # ── Merge ──────────────────────────────────────────────────
    df = skus.merge(
        cv[["item_code", "mean_daily_demand", "std_daily_demand",
            "cv", "cv_class", "adi", "cv2", "demand_pattern", "abc_class"]],
        on="item_code", how="left"
    )

    # Fill SKUs with no orders
    df["mean_daily_demand"] = df["mean_daily_demand"].fillna(0.0)
    df["std_daily_demand"]  = df["std_daily_demand"].fillna(0.0)
    df["cv"]                = df["cv"].fillna(0.0)
    df["cv_class"]          = df["cv_class"].fillna(2).astype(int)
    df["adi"]               = df["adi"].fillna(21.0)
    df["cv2"]               = df["cv2"].fillna(0.0)
    df["demand_pattern"]    = df["demand_pattern"].fillna("intermittent")
    df["abc_class"]         = df["abc_class"].fillna("C")

    # ── Rename abc_class → item_class ─────────────────────────
    df = df.rename(columns={"abc_class": "item_class"})

    # ── Initial inventory ──────────────────────────────────────
    def compute_initial_qty(row):
        mu    = float(row["mean_daily_demand"])
        sigma = float(row["std_daily_demand"])
        cls   = str(row["item_class"])
        cv_c  = int(row["cv_class"])
        pat   = str(row["demand_pattern"])
        cov   = COVERAGE_DAYS.get((cls, pat), COVERAGE_DAYS.get(("C", pat), 4))
        z     = Z_MAP.get((cls, cv_c), norm.ppf(0.80))
        if mu <= 0:
            return 1
        rop_1day = mu * 1.0 + z * sigma * np.sqrt(1.0)
        qty = rop_1day * cov
        return max(1, int(np.ceil(qty)))

    df["item_initial_quantity_inventory"] = df.apply(compute_initial_qty, axis=1)

    # ── ROP global ─────────────────────────────────────────────
    def compute_rop(row):
        mu    = float(row["mean_daily_demand"])
        sigma = float(row["std_daily_demand"])
        cls   = str(row["item_class"])
        cv_c  = int(row["cv_class"])
        z     = Z_MAP.get((cls, cv_c), norm.ppf(0.80))
        if mu <= 0:
            return 1
        rop = mu * LEAD_TIME_ROP + z * sigma * np.sqrt(LEAD_TIME_ROP)
        return max(1, int(np.round(rop)))

    df["rop_global"] = df.apply(compute_rop, axis=1)

    # ── slots_needed = ceil(initial_qty / max_qty_per_slot) ────
    df["max_qty_per_slot"] = (
        np.floor(POD3_SLOT_VOL / df["box_volume"]) * df["number_of_item_in_a_box"]
    ).clip(lower=1)
    df["slots_needed"] = np.ceil(
        df["item_initial_quantity_inventory"] / df["max_qty_per_slot"]
    ).astype(int)

    # ── Final column order ─────────────────────────────────────
    final_cols = [
        "item_code", "item_name",
        "box_length", "box_width", "box_height", "box_volume",
        "box_weight", "number_of_item_in_a_box", "item_volume", "item_unit",
        "item_order_frequency", "item_relative_importance",
        "item_quantity_order_mean", "item_quantity_order_std",
        "item_quantity_order_total", "item_quantity_order_unique",
        "item_quantity_order_unique_frequency",
        "item_class",
        "item_initial_quantity_inventory",
        "mean_daily_demand", "std_daily_demand", "cv",
        "adi", "cv2", "demand_pattern", "cv_class",
        "rop_global", "slots_needed",
    ]
    df = df[final_cols].sort_values("item_order_frequency", ascending=False).reset_index(drop=True)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n  ABC distribution:")
    for cls in ["A", "B", "C"]:
        n = (df["item_class"] == cls).sum()
        print(f"    {cls}: {n:,} ({n/len(df)*100:.1f}%)")

    print(f"\n  Demand pattern:")
    for pat in ["smooth", "erratic", "intermittent", "lumpy"]:
        n = (df["demand_pattern"] == pat).sum()
        print(f"    {pat:13s}: {n:,}")

    print(f"\n  Initial inventory stats:")
    print(f"    min={df['item_initial_quantity_inventory'].min()}, "
          f"max={df['item_initial_quantity_inventory'].max()}, "
          f"mean={df['item_initial_quantity_inventory'].mean():.1f}, "
          f"median={df['item_initial_quantity_inventory'].median():.0f}")

    print(f"\n  ROP stats:")
    print(f"    min={df['rop_global'].min()}, "
          f"max={df['rop_global'].max()}, "
          f"mean={df['rop_global'].mean():.1f}")

    total_slots = df["slots_needed"].sum()
    pods_needed = int(np.ceil(total_slots / 20))
    print(f"\n  Total slots needed  : {total_slots:,}")
    print(f"  Pods needed (20/pod): {pods_needed:,}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved → {OUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
