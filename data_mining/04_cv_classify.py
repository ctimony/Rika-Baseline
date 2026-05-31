"""
04_cv_classify.py
=================
Classifies SKUs by demand pattern and CV class.

Demand pattern (Syntetos, Boylan & Croston, 2005):
  ADI = N_days / n_nonzero_days
  CV² = (std / mean)² of nonzero demand days only
  Cutoffs: ADI=1.32, CV²=0.49
    Smooth       : ADI < 1.32, CV² < 0.49
    Erratic      : ADI < 1.32, CV² >= 0.49
    Intermittent : ADI >= 1.32, CV² < 0.49
    Lumpy        : ADI >= 1.32, CV² >= 0.49

CV class (thesis):
  CV = std_daily / mean_daily  (all days including zeros)
  0 : CV <= 0.5   (Stable)
  1 : 0.5 < CV <= 1.0  (Volatile)
  2 : CV > 1.0   (Intermittent)

Output: data_mining/output/04_cv_classification.csv
"""

import os
import numpy as np
import pandas as pd

ROOT        = os.path.dirname(os.path.abspath(__file__))
ORDERS_PATH = os.path.join(ROOT, "output", "02_clean_orders.csv")
SKUS_PATH   = os.path.join(ROOT, "output", "01_clean_skus_pod3.csv")
ABC_PATH    = os.path.join(ROOT, "output", "03_abc_skus.csv")
OUTPUT_PATH = os.path.join(ROOT, "output", "04_cv_classification.csv")

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


def main():
    print("=" * 60)
    print("  CV Classification & Demand Pattern")
    print("=" * 60)

    orders = pd.read_csv(ORDERS_PATH)
    skus   = pd.read_csv(SKUS_PATH)
    abc    = pd.read_csv(ABC_PATH)

    orders["item_code"] = orders["item_code"].astype(str)
    skus["item_code"]   = skus["item_code"].astype(str)
    abc["item_code"]    = abc["item_code"].astype(str)

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["date"]       = orders["order_date"].dt.date
    all_dates = sorted(orders["date"].unique())
    N = len(all_dates)
    print(f"\n  Observation period  : {N} days ({all_dates[0]} to {all_dates[-1]})")

    # ── Daily demand per SKU (all days, zeros included) ────────
    daily = orders.groupby(["item_code", "date"])["item_quantity"].sum().reset_index()
    all_items = skus["item_code"].unique()
    full_idx  = pd.MultiIndex.from_product(
        [all_items, all_dates], names=["item_code", "date"]
    )
    daily_full = (
        daily.set_index(["item_code", "date"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )

    # ── Compute stats per SKU ──────────────────────────────────
    results = []
    for sku, grp in daily_full.groupby("item_code"):
        vals   = grp["item_quantity"].values.astype(float)
        nonzero = vals[vals > 0]
        n_nz   = len(nonzero)

        # Mean and std over all days (including zeros) — for CV class
        mean_d = vals.mean()
        std_d  = vals.std(ddof=1) if N > 1 else 0.0
        cv     = (std_d / mean_d) if mean_d > 0 else 0.0

        # ADI and CV² from nonzero days only — for demand pattern
        if n_nz == 0:
            adi = float(N)
            cv2 = 0.0
        else:
            adi    = N / n_nz
            mu_nz  = nonzero.mean()
            s_nz   = nonzero.std(ddof=1) if n_nz > 1 else 0.0
            cv2    = (s_nz / mu_nz) ** 2 if mu_nz > 0 else 0.0

        # Demand pattern classification (Syntetos et al. 2005)
        if adi >= ADI_CUTOFF and cv2 >= CV2_CUTOFF:
            pattern = "lumpy"
        elif adi >= ADI_CUTOFF and cv2 < CV2_CUTOFF:
            pattern = "intermittent"
        elif adi < ADI_CUTOFF and cv2 >= CV2_CUTOFF:
            pattern = "erratic"
        else:
            pattern = "smooth"

        # CV class (thesis)
        if cv <= 0.5:
            cv_class = 0
        elif cv <= 1.0:
            cv_class = 1
        else:
            cv_class = 2

        results.append({
            "item_code":        sku,
            "mean_daily_demand": round(mean_d, 4),
            "std_daily_demand":  round(std_d,  4),
            "cv":               round(cv,  4),
            "cv_class":         cv_class,
            "adi":              round(adi, 4),
            "cv2":              round(cv2, 4),
            "demand_pattern":   pattern,
        })

    df = pd.DataFrame(results)

    # ── Merge ABC class ────────────────────────────────────────
    df = df.merge(abc[["item_code", "abc_class"]], on="item_code", how="left")
    df["abc_class"] = df["abc_class"].fillna("C")

    # ── Summary ────────────────────────────────────────────────
    print("\n  Demand pattern distribution:")
    for pat in ["smooth", "erratic", "intermittent", "lumpy"]:
        n = (df["demand_pattern"] == pat).sum()
        print(f"    {pat:13s}: {n:,} ({n/len(df)*100:.1f}%)")

    print("\n  CV class distribution:")
    for cls, label in [(0, "Stable"), (1, "Volatile"), (2, "Intermittent")]:
        n = (df["cv_class"] == cls).sum()
        print(f"    CV{cls} {label:13s}: {n:,} ({n/len(df)*100:.1f}%)")

    print("\n  ABC × CV class:")
    for abc_cls in ["A", "B", "C"]:
        for cv_cls in [0, 1, 2]:
            n = ((df["abc_class"] == abc_cls) & (df["cv_class"] == cv_cls)).sum()
            if n > 0:
                print(f"    {abc_cls} × CV{cv_cls}: {n:,}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
