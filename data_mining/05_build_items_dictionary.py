"""
05_build_items_dictionary.py
============================
Rebuilds items_dictionary.csv from clean pipeline outputs.

Initial inventory & ROP — distribution & horizon by demand pattern.
============================================================================
Both initial_qty and rop_global are built from the SAME per-pattern
distribution; they differ only in the COVERAGE HORIZON. This keeps the two
quantities methodologically consistent (no SKU is born already below its ROP).

Initial inventory covers ONE DAY (24 h) of demand for every pattern; the two
quantities still share the same per-pattern distribution and differ only in the
coverage horizon (24 h for initial, the lead time for ROP).

Smooth / Erratic — Normal lead-time-demand (Pinçe et al. 2021 §3.1;
  Syntetos & Boylan 2005). Demand is frequent, so the CLT holds and the mean
  is a reliable parameter. Statistics in qty-per-hour (mean_hourly, std_hourly
  from 02_clean_orders.csv, zero-filled across all 24 hours).
    initial_qty = ceil( μ_h·H + Z·σ_h·√H ),  H = 24 h (one day)
    rop_global  = round( μ_h·1 + Z·σ_h·√1 ),  L = 1 h

Intermittent / Lumpy — count distribution over BURST SIZE. mean_hourly is
  diluted to ≈0 across 24 h (most hours empty), so it is useless as a base-
  stock parameter. Instead we use the conditional (non-zero) demand statistics
  — the burst size — exactly the µ_i = E[y|y>0], σ²_i = Var(y|y>0) of
  Kronekvist & Titrouq (2026, Eq. 14). The 24-h coverage is converted into the
  EXPECTED NUMBER OF BURSTS that fire in a day, n_act/day = (active SKU-hour
  cells) / (number of days), so the count horizon equals one calendar day:
    z_b   = mean(qty | qty>0)   over active (SKU,date,hour) cells   [burst size]
    s²_b  = var (qty | qty>0)
    n_d   = n_act/day            [expected bursts per day]
    rop_global  = count_ppf(   z_b,    s²_b, SL)   horizon = 1 burst
    initial_qty = count_ppf( n_d·z_b, n_d·s²_b, SL),  horizon = 1 day (n_d bursts),
                  floored at rop_global
  count_ppf fits Negative Binomial when var > mean (overdispersed — "best
  option ... as it allows for greater variability", Pinçe §3.1.3), else Poisson.
  This applies a count distribution DIRECTLY to the burst statistics, avoiding
  the Normal lead-time-demand approximation that Kronekvist & Titrouq (§5.11)
  flag as unsuitable for highly lumpy / intermittent items.

  minimum = 1 everywhere.

Inputs:
  data_mining/output/01_clean_skus_pod3.csv
  data_mining/output/04_cv_classification.csv
  data_mining/output/02_clean_orders.csv

Output: items_dictionary.csv  (project root)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom, poisson

ROOT       = os.path.dirname(os.path.abspath(__file__))
SKUS_PATH  = os.path.join(ROOT, "output", "01_clean_skus_pod3.csv")
CV_PATH    = os.path.join(ROOT, "output", "04_cv_classification.csv")
ORDERS_PATH = os.path.join(ROOT, "output", "02_clean_orders.csv")
OUT_PATH  = os.path.join(ROOT, "..", "items_dictionary.csv")

POD3_SLOT_VOL = 60_000  # cm³

# ── Coverage horizons (per demand pattern) ─────────────────────────────────
# Initial stock covers one calendar day (24 h) for every pattern. Smooth/erratic
# use 24 hours of Normal demand directly; intermittent/lumpy convert the 24-h day
# into n_act/day bursts of burst-size demand. ROP is the lead time (1 h / 1 burst).
INIT_HORIZON_SE_HOURS = 24   # smooth/erratic initial stock = one day
ROP_HORIZON_SE_HOURS  = 1.0    # smooth/erratic ROP lead time = 1 hour
ROP_HORIZON_IL_BURSTS  = 3   # intermittent/lumpy ROP = 1 burst

# Service level (Z) is a function of ABC class ONLY — a proxy for item
# criticality (A items most critical → highest availability), per standard
# inventory practice (Silver, Pyke & Peterson 1998). It is NOT a function of
# cv_class: the legacy cv (total CV including zero-demand days) is misleading for
# intermittent SKUs (high cv from sparsity, not size variability). Demand-size
# variability (CV²) instead enters through the lead-time demand DISTRIBUTION
# (Poisson vs Negative Binomial, chosen by dispersion), per Boylan & Syntetos.
# Service level is set jointly by ABC class AND demand pattern, mirroring the
# (cluster × CV-class) scheme of Chou et al. (Table 4.5). The service level falls
# as demand becomes less regular within a class, because guaranteeing a high
# service level for sporadic demand is uneconomical and rests on an unreliable σ
# estimate (Silver, Pyke & Peterson 1998; Boylan & Syntetos). The four demand
# patterns (smooth/erratic/intermittent/lumpy) map onto Chou's three CV-classes:
#   smooth     → "stable"      (CV0)
#   erratic    → "volatile"    (CV1)
#   intermittent / lumpy → "intermittent" (CV2)
# As in Chou, irregular A items drop to 95%, irregular B items to 90%, and C
# (always intermittent in this dataset) stays at 90% — the floor.
#   A-smooth          : 99%  (Z = 2.3263)   A-volatile/intermittent : 95% / 90%
#   B-regular         : 95%                 B-intermittent          : 90%
#   C                 : 90%
SL_MAP = {
    ("A", "smooth"):       0.99,
    ("A", "erratic"):      0.95,
    ("A", "intermittent"): 0.90,
    ("A", "lumpy"):        0.90,
    ("B", "smooth"):       0.95,
    ("B", "erratic"):      0.95,
    ("B", "intermittent"): 0.90,
    ("B", "lumpy"):        0.90,
    ("C", "smooth"):       0.90,
    ("C", "erratic"):      0.90,
    ("C", "intermittent"): 0.90,
    ("C", "lumpy"):        0.90,
}


def get_service_level(cls, pat):
    """Service level for an (ABC class, demand pattern) pair (Chou Table 4.5).
    Falls back to 0.90 (C floor) for any unclassified combination."""
    return SL_MAP.get((str(cls), str(pat)), 0.90)


def get_z(cls, pat):
    """Safety-stock multiplier Z = Φ⁻¹(service level) for the (class, pattern)."""
    return norm.ppf(get_service_level(cls, pat))

def count_ppf(mu, var, sl):
    """Service-level quantile of a count distribution fitted by dispersion.

    Negative Binomial (method of moments) when overdispersed (var > mu), else
    Poisson (Pinçe et al. 2021 §3.1.3). Used for intermittent/lumpy demand,
    where mu/var are the burst-size statistics scaled by the burst horizon.
    """
    if mu <= 0:
        return 1
    if var > mu:
        r = mu ** 2 / (var - mu)
        p = r / (r + mu)
        q = nbinom.ppf(sl, n=r, p=p)
    else:
        q = poisson.ppf(sl, mu=mu)
    if not np.isfinite(q):
        return 1
    return max(1, int(q))


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

    # ── Recompute item_order_frequency from 02_clean_orders ────
    # Previously item_order_frequency came from the clean-SKU stage (before the
    # quantity/volume filters in 02), so it was inconsistent with
    # mean_daily_demand (which is computed from 02_clean_orders). Recompute it
    # here from the same source = number of unique orders containing the SKU,
    # so the order generator (which samples SKUs ∝ item_order_frequency) and the
    # ROP (∝ mean_daily_demand) are derived from the identical cleaned dataset.
    orders = pd.read_csv(ORDERS_PATH)
    orders["item_code"] = orders["item_code"].astype(str)
    order_freq = orders.groupby("item_code")["order_id"].nunique()
    df["item_order_frequency"] = (
        df["item_code"].map(order_freq).fillna(0).astype(int)
    )

    # ── Hourly demand statistics for ROP ──────────────────────
    # Aggregate qty per (SKU, date, hour), zero-fill hours with no transactions,
    # then compute mean and std across all (date, hour) observations per SKU.
    # This gives demand statistics in units of "qty per hour", consistent with
    # LEAD_TIME_ROP = 1 hour.
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["date"] = orders["order_date"].dt.date
    orders["hour"] = orders["order_date"].dt.hour
    all_dates = sorted(orders["date"].unique())
    all_hours = list(range(24))
    all_order_skus = orders["item_code"].unique()
    hourly_qty = (
        orders.groupby(["item_code", "date", "hour"])["item_quantity"].sum()
        .reindex(
            pd.MultiIndex.from_product(
                [all_order_skus, all_dates, all_hours],
                names=["item_code", "date", "hour"],
            ),
            fill_value=0,
        )
    )
    mean_hourly_series = hourly_qty.groupby("item_code").mean()
    std_hourly_series  = hourly_qty.groupby("item_code").std(ddof=1).fillna(0)
    df["mean_hourly_demand"] = df["item_code"].map(mean_hourly_series).fillna(0.0)
    df["std_hourly_demand"]  = df["item_code"].map(std_hourly_series).fillna(0.0)

    # ── Burst-size statistics (intermittent / lumpy) ──────────────
    # Conditional (non-zero) demand statistics over active (SKU,date,hour)
    # cells: z_b = E[qty | qty>0] (burst size), s2_b = Var[qty | qty>0].
    # These are the µ_i, σ²_i of Kronekvist & Titrouq (2026, Eq. 14). For
    # intermittent/lumpy SKUs the diluted mean_hourly is ≈0, so the burst
    # statistics — not the hourly mean — drive the count-distribution ROP and
    # initial stock (horizon measured in bursts).
    active = (
        orders.groupby(["item_code", "date", "hour"])["item_quantity"].sum()
        .reset_index()
    )
    zb_series  = active.groupby("item_code")["item_quantity"].mean()
    s2b_series = active.groupby("item_code")["item_quantity"].var(ddof=1).fillna(0.0)
    df["burst_size"]     = df["item_code"].map(zb_series).fillna(0.0)
    df["burst_size_var"] = df["item_code"].map(s2b_series).fillna(0.0)

    # Expected bursts per day, n_act/day = (active SKU-hour cells) / (#days).
    # Used as the count horizon for intermittent/lumpy initial stock (one day).
    n_days = max(1, len(all_dates))
    n_act_series = active.groupby("item_code")["item_quantity"].size() / n_days
    df["n_act_per_day"] = df["item_code"].map(n_act_series).fillna(0.0)

    # ── ROP global ─────────────────────────────────────────────
    # Same per-pattern distribution as initial stock; horizon = 1 (hour or
    # burst). See module docstring.
    #   Smooth / Erratic   : Normal,  rop = μ_h·L + Z·σ_h·√L,  L = 1 h
    #   Intermittent/Lumpy : count_ppf over burst size, horizon = 1 burst
    def compute_rop(row):
        pat = str(row["demand_pattern"])
        cls = str(row["item_class"])
        sl  = get_service_level(cls, pat)

        if pat in ("smooth", "erratic"):
            mu_h  = float(row["mean_hourly_demand"])
            sig_h = float(row["std_hourly_demand"])
            if mu_h <= 0:
                return 1
            L   = ROP_HORIZON_SE_HOURS
            rop = mu_h * L + norm.ppf(sl) * sig_h * np.sqrt(L)
            return max(1, int(np.round(rop)))

        # Intermittent / Lumpy — count distribution over ONE full demand event
        # (1 burst). Because replenishment is TRIGGERED by a demand occurrence, the
        # ROP must cover at least one full demand event (Teunter & Duncan 2009); a
        # sub-event ROP is meaningless for sporadic demand whose per-hour rate ≈ 0.
        # ROP_IL is therefore FIXED at one burst (independent of lead time): the
        # lead-time knob (ROP_HORIZON_SE_HOURS) varies the SE reorder point only,
        # since SE demand is regular enough for an hours-based lead time to be
        # meaningful, whereas IL demand is event-based. This keeps the IL reorder
        # rule simple and standard, and (as the trigger log shows) SE SKUs drive the
        # large majority of replenishments, so the lead-time lever remains effective.
        return count_ppf(float(row["burst_size"]),
                         float(row["burst_size_var"]), sl)

    df["rop_global"] = df.apply(compute_rop, axis=1)

    # ── Initial inventory ──────────────────────────────────────
    # One day (24 h) of coverage, same per-pattern distribution as ROP.
    # NOT floored at ROP: initial stock is a fixed 24-h coverage that must stay
    # CONSTANT when the ROP lead-time horizon (ROP_HORIZON_SE_HOURS / _IL_BURSTS)
    # is varied for the DoE. Flooring at ROP would let a higher lead-time ROP drag
    # initial stock (and hence slots_needed / pod allocation) up — coupling the
    # lead-time knob to stock & layout. We want lead time to move ROP (the reorder
    # TRIGGER) ONLY, so the initial stock / pod layout are identical across lead-time
    # scenarios and the comparison isolates the trigger.
    #   Smooth / Erratic   : Normal over H = 24 h (one day)
    #   Intermittent/Lumpy : count_ppf over n_act/day bursts of burst size (one day)
    def compute_initial_qty(row):
        pat = str(row["demand_pattern"])
        cls = str(row["item_class"])
        sl  = get_service_level(cls, pat)

        if pat in ("smooth", "erratic"):
            mu_h  = float(row["mean_hourly_demand"])
            sig_h = float(row["std_hourly_demand"])
            if mu_h <= 0:
                return 1
            H   = INIT_HORIZON_SE_HOURS
            qty = mu_h * H + norm.ppf(sl) * sig_h * np.sqrt(H)
            return max(int(np.ceil(qty)), 1)

        # Intermittent / Lumpy — count distribution over one day = n_act/day bursts.
        # At least one burst of coverage even if n_act/day rounds toward zero.
        k   = max(1.0, float(row["n_act_per_day"]))
        qty = count_ppf(k * float(row["burst_size"]),
                        k * float(row["burst_size_var"]), sl)
        return max(qty, 1)

    df["item_initial_quantity_inventory"] = df.apply(compute_initial_qty, axis=1)

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
        "mean_hourly_demand", "std_hourly_demand",
        "burst_size", "burst_size_var", "n_act_per_day",
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
