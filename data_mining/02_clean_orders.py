"""
02_clean_orders.py
==================
Cleans raw_order.csv into clean orders. This script runs FIRST in the cleaning
stage; 01_clean_skus_pod3.py runs AFTER and derives the clean SKU master from
the output of this script (only SKUs that actually appear in clean_orders are
kept).

Master dimensions come from `items_dictionary old.csv` (the most complete SKU
master, 7,442 SKUs).

Cleaning steps:
1. drop_duplicates() — remove fully identical rows (same order_id + item_code +
   item_quantity + order_date = data-entry duplicates → keep one)
2. Consolidate remaining duplicate rows (same order_id + item_code, qty differs)
   by summing item_quantity — treated as genuine multi-line orders
3. Remove null values
4. Remove order lines whose item_volume > pod type 3 slot volume (60,000 cm³)
   — a SKU that does not fit a slot is removed entirely, so it will also be
   absent from clean_skus downstream.

NOTE: the old "item_quantity > max_qty_per_slot" filter has been REMOVED.
Large-quantity orders are kept; the elevated demand they produce flows into
mean_daily_demand and is absorbed by slots_needed (= ceil(initial_qty /
max_qty_per_slot)) downstream — capacity adapts to demand instead of discarding
real orders.

Input : raw_order.csv, items_dictionary old.csv
Output: data_mining/output/02_clean_orders.csv
"""

import os
import numpy as np
import pandas as pd

ROOT            = os.path.dirname(os.path.abspath(__file__))
RAW_ORDER_PATH  = os.path.join(ROOT, "..", "raw_order.csv")
MASTER_PATH     = os.path.join(ROOT, "..", "items_dictionary old.csv")
OUTPUT_PATH     = os.path.join(ROOT, "output", "02_clean_orders.csv")
REPORT_PATH     = os.path.join(ROOT, "output", "02_cleaning_report.txt")

POD3_SLOT_VOL = 60_000  # cm³ — slot volume for pod type 3 (20 slots/pod)


def main():
    print("=" * 60)
    print("  Clean orders — raw_order.csv (pod type 3, slot vol 60,000 cm³)")
    print("=" * 60)

    raw    = pd.read_csv(RAW_ORDER_PATH)
    master = pd.read_csv(MASTER_PATH)

    raw.columns    = raw.columns.str.lstrip('﻿')
    master.columns = master.columns.str.lstrip('﻿')
    raw["item_code"]    = raw["item_code"].astype(str)
    master["item_code"] = master["item_code"].astype(str)

    n_raw = len(raw)
    print(f"\n  Raw orders loaded   : {n_raw:,} rows")
    report_lines = [f"Raw orders: {n_raw:,}"]

    # ── Step 1: Drop fully identical rows (data-entry duplicates) ──
    raw = raw.drop_duplicates()
    n1 = len(raw)
    print(f"  After drop identical: {n1:,} rows  (removed {n_raw - n1:,} exact dupes)")
    report_lines.append(f"After drop identical rows: {n1:,} (removed {n_raw - n1:,})")

    # ── Step 2: Remove nulls ───────────────────────────────────
    raw = raw.dropna()
    n2 = len(raw)
    print(f"  After drop nulls    : {n2:,} rows  (removed {n1 - n2:,})")
    report_lines.append(f"After drop nulls: {n2:,} (removed {n1 - n2:,})")

    # ── Step 3: Consolidate remaining duplicates by summing qty ──
    # (same order_id + item_code with DIFFERING quantity → genuine multi-line)
    meta = raw.drop_duplicates(subset=["order_id", "item_code"])[
        ["order_id", "item_code", "item_name", "order_date"]
    ]
    qty_sum = raw.groupby(["order_id", "item_code"], as_index=False)["item_quantity"].sum()
    raw = meta.merge(qty_sum, on=["order_id", "item_code"])
    n3 = len(raw)
    print(f"  After consolidate   : {n3:,} rows  (merged {n2 - n3:,} multi-line)")
    report_lines.append(f"After consolidate duplicates: {n3:,} (merged {n2 - n3:,})")

    # ── Step 4: Remove items that do not fit a slot ──────────────
    # A SKU is dropped (all its order lines) if EITHER:
    #   - item_volume > slot volume  → a single unit does not fit, OR
    #   - box_volume  > slot volume  → the box does not fit, so
    #     max_qty_per_slot = floor(slot_vol/box_vol)*n = 0 (cannot be stored).
    # Both checks are required: a SKU can have a small item_volume but an
    # oversized box (e.g. bulky tissue), which would otherwise yield a
    # divide-by-zero max_qty_per_slot and an inflated slots_needed downstream.
    vols = master[["item_code", "item_volume", "box_volume"]].copy()
    raw = raw.merge(vols, on="item_code", how="left")
    too_large = (
        raw["item_volume"].isna()
        | (raw["item_volume"] > POD3_SLOT_VOL)
        | (raw["box_volume"] > POD3_SLOT_VOL)
    )
    raw = raw[~too_large].copy()
    n4 = len(raw)
    print(f"  After vol filter    : {n4:,} rows  (removed {n3 - n4:,})")
    report_lines.append(f"After item/box_volume > {POD3_SLOT_VOL:,} filter: {n4:,} (removed {n3 - n4:,})")

    # ── Step 5: Remove order lines where qty > max_qty_per_slot ──────
    # max_qty_per_slot = floor(slot_volume / box_volume) * number_of_item_in_a_box
    # An order asking for more than one slot can hold is dropped (1-slot model).
    # (box_volume is already merged from Step 4; only add number_of_item_in_a_box.)
    nbox = master[["item_code", "number_of_item_in_a_box"]].copy()
    raw = raw.merge(nbox, on="item_code", how="left")
    raw["max_qty_per_slot"] = (
        np.floor(POD3_SLOT_VOL / raw["box_volume"]) * raw["number_of_item_in_a_box"]
    )
    qty_exceed = (
        raw["max_qty_per_slot"].notna()
        & (raw["item_quantity"] > raw["max_qty_per_slot"])
    )
    raw = raw[~qty_exceed].copy()
    n5 = len(raw)
    print(f"  After qty cap filter: {n5:,} rows  (removed {n4 - n5:,})")
    report_lines.append(f"After qty > max_qty_per_slot filter: {n5:,} (removed {n4 - n5:,})")

    # ── Finalize ────────────────────────────────────────────────
    raw = raw[["order_id", "item_code", "item_name", "item_quantity", "order_date"]].copy()
    raw = raw.sort_values(["order_date", "order_id", "item_code"]).reset_index(drop=True)

    print(f"\n  Final clean orders  : {len(raw):,} rows")
    print(f"  Unique orders       : {raw['order_id'].nunique():,}")
    print(f"  Unique items        : {raw['item_code'].nunique():,}")
    print(f"  Total removed       : {n_raw - len(raw):,} ({(n_raw - len(raw))/n_raw*100:.1f}%)")

    report_lines += [
        f"\nFinal clean orders: {len(raw):,}",
        f"Unique orders: {raw['order_id'].nunique():,}",
        f"Unique items: {raw['item_code'].nunique():,}",
        f"Total removed: {n_raw - len(raw):,} ({(n_raw - len(raw))/n_raw*100:.1f}%)",
    ]

    raw.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report → {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
