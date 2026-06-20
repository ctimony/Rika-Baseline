"""
01_clean_skus_pod3.py
=====================
Builds the clean SKU master. This script runs AFTER 02_clean_orders.py.

The clean SKU master = `items_dictionary old.csv` (the most complete master,
7,442 SKUs) filtered to ONLY the SKUs that actually appear in 02_clean_orders.csv.

Rationale: clean_orders is the ground truth of what was transacted. A SKU that is
never ordered does not belong in the storage master (it would otherwise sit as an
inert "ghost" SKU with zero demand, occupying a slot). Because clean_orders has
already removed SKUs whose item_volume > slot volume (60,000 cm³), those oversized
SKUs are automatically excluded here too — no separate volume filter is needed.

Input : items_dictionary old.csv, data_mining/output/02_clean_orders.csv
Output: data_mining/output/01_clean_skus_pod3.csv
"""

import os
import pandas as pd

ROOT         = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH  = os.path.join(ROOT, "..", "items_dictionary old.csv")
ORDERS_PATH  = os.path.join(ROOT, "output", "02_clean_orders.csv")
OUTPUT_PATH  = os.path.join(ROOT, "output", "01_clean_skus_pod3.csv")


def main():
    print("=" * 60)
    print("  Clean SKUs — master filtered to ordered SKUs")
    print("=" * 60)

    master = pd.read_csv(MASTER_PATH)
    orders = pd.read_csv(ORDERS_PATH)

    master.columns = master.columns.str.lstrip('﻿')
    master["item_code"] = master["item_code"].astype(str)
    orders["item_code"] = orders["item_code"].astype(str)

    n_master = len(master)
    ordered_skus = set(orders["item_code"].unique())
    print(f"\n  Master SKUs (old)       : {n_master:,}")
    print(f"  Distinct SKUs in orders : {len(ordered_skus):,}")

    # Keep only SKUs that appear in clean_orders
    df = master[master["item_code"].isin(ordered_skus)].copy().reset_index(drop=True)
    n_kept = len(df)
    print(f"\n  Kept (ordered) SKUs     : {n_kept:,}  (removed {n_master - n_kept:,} never-ordered)")

    # Sanity: every kept SKU is ordered, and every ordered SKU is in master
    missing = ordered_skus - set(df["item_code"])
    if missing:
        print(f"  WARNING: {len(missing)} ordered SKUs not found in master "
              f"(e.g. {list(missing)[:5]})")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
