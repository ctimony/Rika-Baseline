"""
06_sampling.py
==============
Impact-score quartile sampling of SKUs for simulation (Opsi A+).

Why sample at all:
  The full dictionary has 7,081 SKUs needing ~13,026 pod slots, but the
  physical grid holds 500 pod cells, 10 of which are charging stations →
  489 active SKU-storage pods × 20 slots = 9,780 slot budget. So a subset must
  be chosen. This is also operationally realistic for an RMFS: not every
  slow-mover (class C) is kept online in mobile pods at once.

Selection method — impact-quartile priority fill (deterministic):
  1. impact score per SKU = geometric mean of 3 factors
       composite   = mean_daily_demand * item_order_frequency
       stability   = 1 / ADI                  (regular demand ↑)
       variability = 1 + CV²                  (size variability)
       impact = (composite * stability * variability)^(1/3)
  2. Within each ABC class, split SKUs into 4 quartiles by impact (Q4=highest).
  3. ALL of class A is taken IN FULL — the high-movers are never sampled away
     (they fit in ~half the pod budget).
  4. PRIORITY FILL by quartile: from the remaining budget, take all of the
     highest quartile first (Q4 of B and C), then Q3, then Q2, then Q1, until
     the slot budget is exhausted. Whatever quartile the budget runs out in is
     filled top-impact-first; lower quartiles below it are dropped. This GUARANTEES
     the highest-impact SKUs (Q4, and as far down as the budget reaches) are kept
     in full — drops fall only on the lowest-impact quartile(s).

Why priority fill (vs uniform fraction per quartile):
  - Taking all of class A + all of Q4 means NO high-impact SKU is ever dropped.
  - The drop is concentrated on the least-impactful tail (lowest quartile), which
    is operationally what an RMFS would keep offline — instead of shaving a uniform
    slice off every quartile (which would discard some high-impact Q4 SKUs).

Output: data_mining/output/06_sampled_skus.csv
"""

import os
import numpy as np
import pandas as pd

ROOT        = os.path.dirname(os.path.abspath(__file__))
DICT_PATH   = os.path.join(ROOT, "..", "items_dictionary.csv")
OUTPUT_PATH = os.path.join(ROOT, "output", "06_sampled_skus.csv")

# Pod budget = active SKU pods (layout.py total_pods_active) × slots per pod.
ACTIVE_PODS    = 489
SLOTS_PER_POD  = 20
SLOT_BUDGET    = ACTIVE_PODS * SLOTS_PER_POD   # 9,780
PROTECT_CLASSES = ["A"]                         # ABC classes taken in full (never sampled)


def main():
    print("=" * 60)
    print("  SKU Sampling — Impact-Quartile Stratified (Opsi A+)")
    print("=" * 60)

    df = pd.read_csv(DICT_PATH)
    df["item_code"] = df["item_code"].astype(str)
    print(f"\n  Input SKUs    : {len(df):,}")
    print(f"  Slot budget   : {SLOT_BUDGET:,} ({ACTIVE_PODS} pods × {SLOTS_PER_POD})")

    # ── Impact score ───────────────────────────────────────────
    df["composite"]   = df["mean_daily_demand"] * df["item_order_frequency"]
    df["stability"]   = 1.0 / df["adi"].clip(lower=0.01)
    df["variability"] = 1.0 + df["cv2"]
    df["impact"] = (
        (df["composite"] * df["stability"] * df["variability"]).clip(lower=0).pow(1 / 3)
    )

    # ── Quartile within each ABC class ─────────────────────────
    df["quartile"] = df.groupby("item_class")["impact"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    )

    classes = ["A", "B", "C"]
    # Quartiles from highest impact (Q4) to lowest (Q1) — fill order.
    quarts_hi_to_lo = ["Q4", "Q3", "Q2", "Q1"]

    # ── Priority fill ──────────────────────────────────────────
    # 1) Protected classes (A) taken in full, regardless of budget.
    # 2) Remaining budget filled quartile-by-quartile, highest impact first,
    #    each quartile (B+C pooled) taken top-impact-first. The quartile where
    #    the budget runs out is partially filled; lower quartiles are dropped.
    protected = df[df["item_class"].isin(PROTECT_CLASSES)]
    samplable = df[~df["item_class"].isin(PROTECT_CLASSES)]

    parts = [protected]
    used  = int(protected["slots_needed"].sum())
    cutoff_quartile = None

    for q in quarts_hi_to_lo:
        cell = samplable[samplable["quartile"] == q].sort_values(
            "impact", ascending=False
        )
        cum = cell["slots_needed"].cumsum()
        fits = cell[used + cum <= SLOT_BUDGET]
        parts.append(fits)
        used += int(fits["slots_needed"].sum())
        if len(fits) < len(cell):
            cutoff_quartile = q
            break  # budget exhausted within this quartile; lower quartiles dropped

    result = pd.concat(parts)
    total_slots = int(result["slots_needed"].sum())
    pods_used   = int(np.ceil(total_slots / SLOTS_PER_POD))

    # ── Summary ────────────────────────────────────────────────
    print(f"\n  Priority fill by quartile (Q4→Q1, highest impact first)")
    print(f"  Class A protected (taken in full)")
    if cutoff_quartile is not None:
        print(f"  Budget ran out in quartile: {cutoff_quartile} "
              f"(quartiles below it fully dropped)")
    else:
        print(f"  All quartiles fully kept (budget not binding)")
    print(f"\n  Selected SKUs : {len(result):,}")
    print(f"  Slots used    : {total_slots:,} / {SLOT_BUDGET:,}")
    print(f"  Pods used     : {pods_used:,} / {ACTIVE_PODS:,}")

    print(f"\n  ABC stock mix (selected vs population):")
    for cls in classes:
        sel_pct = (result["item_class"] == cls).mean() * 100
        pop_pct = (df["item_class"] == cls).mean() * 100
        print(f"    {cls}: {sel_pct:4.1f}%  (population {pop_pct:4.1f}%)")

    print(f"\n  Demand share by order_frequency (selected vs population):")
    fsel = result.groupby("item_class")["item_order_frequency"].sum()
    fpop = df.groupby("item_class")["item_order_frequency"].sum()
    for cls in classes:
        print(f"    {cls}: {fsel[cls]/fsel.sum():.3f}  (population {fpop[cls]/fpop.sum():.3f})")
    cover = result["mean_daily_demand"].sum() / df["mean_daily_demand"].sum() * 100
    print(f"\n  Demand coverage : {cover:.1f}%")

    print(f"\n  Class A coverage (must be 100% — A taken in full):")
    A = df[df["item_class"] == "A"]
    A_used = A["item_code"].isin(set(result["item_code"]))
    print(f"    A used: {A_used.sum():,}/{len(A):,}")

    # ── Save (drop helper columns) ─────────────────────────────
    drop_cols = ["composite", "stability", "variability", "impact", "quartile"]
    out = result.drop(columns=[c for c in drop_cols if c in result.columns])
    out = out.sort_values("item_order_frequency", ascending=False).reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
