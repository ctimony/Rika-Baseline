"""
05_build_items_dictionary.py
============================
Rebuilds items_dictionary.csv from clean pipeline outputs.

Initial inventory formula:
  initial_qty = mean_daily_demand * cov_days + Z * std_daily_demand * sqrt(cov_days)

Coverage days per ABC × demand pattern:
  A: smooth/erratic=1, intermittent=3, lumpy=4
  B: smooth/erratic=2, intermittent=4, lumpy=5
  C: smooth/erratic=3, intermittent=5, lumpy=5

Z score per ABC × CV class:
  A × CV0 : 2.33 (99%)    B × CV0 : 1.64 (95%)    C × CV0 : 1.28 (90%)
  A × CV1 : 1.64 (95%)    B × CV1 : 1.64 (95%)    C × CV1 : 1.28 (90%)
  A × CV2 : 1.28 (90%)    B × CV2 : 1.28 (90%)    C × CV2 : 1.28 (90%)

ROP — distribution-fitting by demand pattern (lead time L = 1/8 day):
  ROP = F⁻¹(service level), with lead-time demand μ(L)=μ·L, σ²(L)=σ²·L

  Smooth / Erratic — Normal (Pinçe et al. 2021 §3.1; Syntetos & Boylan 2005):
    rop = μ·L + Z · σ·sqrt(L)

  Intermittent / Lumpy — count distribution (Pinçe et al. 2021 §3.1.3):
    Negative Binomial when var(L) > mean(L)  [overdispersed — "best option ...
      as it allows for greater variability"]; method of moments for (r, p).
    Poisson when var(L) ≤ mean(L)  [equidispersed, near-constant demand size].

  minimum ROP = 1

Inputs:
  data_mining/output/01_clean_skus_pod3.csv
  data_mining/output/04_cv_classification.csv

Output: items_dictionary.csv  (project root)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom, poisson

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

# Service level (Z) is a function of ABC class ONLY — a proxy for item
# criticality (A items most critical → highest availability), per standard
# inventory practice (Silver, Pyke & Peterson 1998). It is NOT a function of
# cv_class: the legacy cv (total CV including zero-demand days) is misleading for
# intermittent SKUs (high cv from sparsity, not size variability). Demand-size
# variability (CV²) instead enters through the lead-time demand DISTRIBUTION
# (Poisson vs Negative Binomial, chosen by dispersion), per Boylan & Syntetos.
#   A: 99% service level (Z = 2.3263)
#   B: 95% service level (Z = 1.6449)
#   C: 90% service level (Z = 1.2816)
Z_MAP = {
    "A": norm.ppf(0.99),  # 2.3263
    "B": norm.ppf(0.95),  # 1.6449
    "C": norm.ppf(0.90),  # 1.2816
}

LEAD_TIME_ROP = 0.5 / 8  # 1 shift-hour (default)


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
        z     = Z_MAP.get(cls, Z_MAP["C"])  # fallback: unclassified -> C (90%), consistent with fillna("C")
        if mu <= 0:
            return 1
        rop_1day = mu * 1.0 + z * sigma * np.sqrt(1.0)
        qty = rop_1day * cov
        return max(1, int(np.ceil(qty)))

    df["item_initial_quantity_inventory"] = df.apply(compute_initial_qty, axis=1)

    # ── ROP global ─────────────────────────────────────────────
    # Distribution-fitting approach (Pinçe et al. 2021; Syntetos et al. 2005;
    # Teunter & Duncan 2009). Demand pattern (from ADI & CV²) selects the
    # lead-time demand distribution; ROP = F⁻¹(service level).
    #
    #   Smooth / Erratic   : Normal       — regular demand (Pinçe §3.1)
    #   Intermittent/Lumpy : Negative Binomial when overdispersed (var > mean)
    #                        — "best option is the negative binomial distribution,
    #                        as it allows for greater variability" (Pinçe §3.1.3).
    #                        Poisson when var ≤ mean (equidispersed demand sizes).
    #
    # Lead-time extrapolation (Pinçe §3): μ(L) = μ·L, σ²(L) = σ²·L
    def compute_rop(row):
        mu    = float(row["mean_daily_demand"])
        sigma = float(row["std_daily_demand"])
        pat   = str(row["demand_pattern"])
        cls   = str(row["item_class"])
        cv_c  = int(row["cv_class"])
        z     = Z_MAP.get(cls, Z_MAP["C"])  # fallback: unclassified -> C (90%), consistent with fillna("C")
        sl    = norm.cdf(z)        # service level probability from Z
        L     = LEAD_TIME_ROP

        if mu <= 0:
            return 1

        if pat in ("smooth", "erratic"):
            # Normal distribution — standard formula for regular demand
            rop = mu * L + z * sigma * np.sqrt(L)
            return max(1, int(np.round(rop)))

        # Intermittent / Lumpy — count distribution fitted to lead-time demand
        mu_L  = mu * L                  # mean lead-time demand
        var_L = (sigma ** 2) * L        # variance lead-time demand

        if var_L > mu_L:
            # Negative Binomial via method of moments (overdispersed)
            #   var = μ + μ²/r  →  r = μ²/(var − μ),  p = r/(r + μ)
            r = mu_L ** 2 / (var_L - mu_L)
            p = r / (r + mu_L)
            rop = nbinom.ppf(sl, n=r, p=p)
        else:
            # Poisson (equidispersed: var ≤ mean, near-constant demand size)
            rop = poisson.ppf(sl, mu=mu_L)

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
