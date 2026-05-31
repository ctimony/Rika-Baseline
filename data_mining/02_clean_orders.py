"""
02_clean_orders.py
==================
Cleans raw_order.csv using pod type 3 slot volume (60,000 cm³).

Cleaning steps:
1. Remove null values
2. Consolidate duplicate rows (same order_id + item_code) by summing item_quantity
3. Remove items whose item_volume > pod type 3 slot volume (60,000 cm³)
   — also drops items not found in clean_skus (item_volume will be NaN)
4. Remove order lines where item_quantity > max_qty_per_slot
   max_qty_per_slot = floor(slot_volume / box_volume) * number_of_item_in_a_box
5. Cross-check item_code against 01_clean_skus_pod3.csv

Output: data_mining/output/02_clean_orders.csv
"""

import os
import numpy as np
import pandas as pd

ROOT            = os.path.dirname(os.path.abspath(__file__))
RAW_ORDER_PATH  = os.path.join(ROOT, "..", "raw_order.csv")
CLEAN_SKUS_PATH = os.path.join(ROOT, "output", "01_clean_skus_pod3.csv")
OUTPUT_PATH     = os.path.join(ROOT, "output", "02_clean_orders.csv")
REPORT_PATH     = os.path.join(ROOT, "output", "02_cleaning_report.txt")

POD3_SLOT_VOL = 60_000  # cm³ — slot volume for pod type 3 (20 slots/pod)


def main():
    print("=" * 60)
    print("  Data Cleaning — raw_order.csv (pod type 3)")
    print("=" * 60)

    raw   = pd.read_csv(RAW_ORDER_PATH)
    items = pd.read_csv(CLEAN_SKUS_PATH)

    raw.columns   = raw.columns.str.lstrip('﻿')
    items.columns = items.columns.str.lstrip('﻿')
    raw["item_code"]   = raw["item_code"].astype(str)
    items["item_code"] = items["item_code"].astype(str)

    n_raw = len(raw)
    print(f"\n  Raw orders loaded   : {n_raw:,} rows")
    report_lines = [f"Raw orders: {n_raw:,}"]

    # ── Step 1: Remove nulls ───────────────────────────────────
    raw = raw.dropna()
    n1 = len(raw)
    print(f"  After drop nulls    : {n1:,} rows  (removed {n_raw - n1:,})")
    report_lines.append(f"After drop nulls: {n1:,} (removed {n_raw - n1:,})")

    # ── Step 2: Consolidate duplicates by summing item_quantity ──
    meta = raw.drop_duplicates(subset=["order_id", "item_code"])[
        ["order_id", "item_code", "item_name", "order_date"]
    ]
    qty_sum = raw.groupby(["order_id", "item_code"], as_index=False)["item_quantity"].sum()
    raw = meta.merge(qty_sum, on=["order_id", "item_code"])
    n2 = len(raw)
    print(f"  After consolidate   : {n2:,} rows  (merged {n1 - n2:,} dupes)")
    report_lines.append(f"After consolidate duplicates: {n2:,} (merged {n1 - n2:,} dupes)")

    # ── Step 3: Remove items whose item_volume > pod type 3 slot volume ──
    item_vol = items[["item_code", "item_volume"]].copy()
    raw = raw.merge(item_vol, on="item_code", how="left")
    too_large = raw["item_volume"].isna() | (raw["item_volume"] > POD3_SLOT_VOL)
    raw = raw[~too_large].copy()
    n3 = len(raw)
    print(f"  After vol filter    : {n3:,} rows  (removed {n2 - n3:,})")
    report_lines.append(f"After item_volume > {POD3_SLOT_VOL:,} filter: {n3:,} (removed {n2 - n3:,})")

    # ── Step 4: Remove order lines where qty > max_qty_per_slot ──────
    # max_qty_per_slot = floor(slot_volume / box_volume) * number_of_item_in_a_box
    item_cap = items[["item_code", "box_volume", "number_of_item_in_a_box"]].copy()
    raw = raw.merge(item_cap, on="item_code", how="left")
    raw["max_qty_per_slot"] = (
        np.floor(POD3_SLOT_VOL / raw["box_volume"]) * raw["number_of_item_in_a_box"]
    )
    qty_exceed = (
        raw["max_qty_per_slot"].notna() &
        (raw["item_quantity"] > raw["max_qty_per_slot"])
    )
    raw = raw[~qty_exceed].copy()
    n4 = len(raw)
    print(f"  After qty cap filter: {n4:,} rows  (removed {n3 - n4:,})")
    report_lines.append(f"After qty > max_qty_per_slot filter: {n4:,} (removed {n3 - n4:,})")

    # ── Step 5: Keep only items in clean_skus_pod3 ─────────────
    valid_items = set(items["item_code"])
    raw = raw[raw["item_code"].isin(valid_items)].copy()
    n5 = len(raw)
    print(f"  After dict filter   : {n5:,} rows  (removed {n4 - n5:,})")
    report_lines.append(f"After clean_skus_pod3 filter: {n5:,} (removed {n4 - n5:,})")

    # ── Finalize ────────────────────────────────────────────────
    raw = raw[["order_id", "item_code", "item_name", "item_quantity", "order_date"]].copy()
    raw = raw.sort_values(["order_date", "order_id", "item_code"]).reset_index(drop=True)

    print(f"\n  Final clean orders  : {n5:,} rows")
    print(f"  Unique orders       : {raw['order_id'].nunique():,}")
    print(f"  Unique items        : {raw['item_code'].nunique():,}")
    print(f"  Total removed       : {n_raw - n5:,} ({(n_raw - n5)/n_raw*100:.1f}%)")

    report_lines += [
        f"\nFinal clean orders: {n5:,}",
        f"Unique orders: {raw['order_id'].nunique():,}",
        f"Unique items: {raw['item_code'].nunique():,}",
        f"Total removed: {n_raw - n5:,} ({(n_raw - n5)/n_raw*100:.1f}%)",
    ]

    raw.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report → {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
