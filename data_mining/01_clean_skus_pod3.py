"""
01_clean_skus_pod3.py
=====================
Filters clean_skus for pod type 3 slot volume (60,000 cm³).

Two filters applied:
1. item_volume > 60,000 cm³ — item itself does not fit in a slot
2. box_volume > 60,000 cm³ — box does not fit in a slot (even if item fits,
   the box cannot be stored → max_qty_per_slot = 0 → all orders filtered out)

Input : data_mining/output/00_clean_skus.csv
Output: data_mining/output/01_clean_skus_pod3.csv
"""

import os
import pandas as pd

ROOT        = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(ROOT, "output", "00_clean_skus.csv")
OUTPUT_PATH = os.path.join(ROOT, "output", "01_clean_skus_pod3.csv")

POD3_SLOT_VOL = 60_000  # cm³ — slot volume for pod type 3 (20 slots/pod)


def main():
    print("=" * 60)
    print("  Clean SKUs — pod type 3 filter (slot vol 60,000 cm³)")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)
    n_raw = len(df)
    print(f"\n  Input SKUs          : {n_raw:,}")

    # Filter 1: item_volume > slot volume
    too_large_item = df["item_volume"] > POD3_SLOT_VOL
    removed_item = df[too_large_item][["item_code", "item_name", "item_volume", "box_volume"]]
    if len(removed_item) > 0:
        print(f"\n  SKUs removed (item_volume > {POD3_SLOT_VOL:,} cm³): {len(removed_item)}")
        for _, r in removed_item.iterrows():
            print(f"    {r['item_code']} — {r['item_name']} — item_vol={r['item_volume']:,.0f} cm³")

    df = df[~too_large_item].copy().reset_index(drop=True)
    n1 = len(df)
    print(f"\n  After item_volume filter : {n1:,}  (removed {n_raw - n1:,})")

    # Filter 2: box_volume > slot volume (box doesn't fit in slot)
    too_large_box = df["box_volume"] > POD3_SLOT_VOL
    removed_box = df[too_large_box][["item_code", "item_name", "item_volume", "box_volume"]]
    if len(removed_box) > 0:
        print(f"\n  SKUs removed (box_volume > {POD3_SLOT_VOL:,} cm³): {len(removed_box)}")
        for _, r in removed_box.iterrows():
            print(f"    {r['item_code']} — {r['item_name']} — box_vol={r['box_volume']:,.0f} cm³")

    df = df[~too_large_box].copy().reset_index(drop=True)
    n2 = len(df)
    print(f"\n  After box_volume filter  : {n2:,}  (removed {n1 - n2:,})")
    print(f"  Total removed            : {n_raw - n2:,}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved → {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
