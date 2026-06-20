"""
00_data_cleaning.py — Data Cleaning & Routing

Steps:
  1. Load raw_order.csv, items_dictionary.csv, and items_slots_configuration.csv
  2. Remove duplicates and null values from both datasets
  3. Filter SKUs that do not fit into any pod slot type using items_slots_configuration.csv
     (max_box_in_slot == 0 for all slot types → truly oversized, redirected to big-item facility)
  4. Identify SKUs with exceptionally high single-order quantities (> 500 units)
     → these are written to a separate file (redirected to split-order warehouse)
  5. Apply both filters to produce a clean order dataset and a clean SKU master

Run: python data_mining/00_data_cleaning.py
"""

import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_ORDER_PATH   = os.path.join(os.path.dirname(__file__), "..", "raw_order.csv")
ITEMS_DICT_PATH  = os.path.join(os.path.dirname(__file__), "..", "items_dictionary.csv")
ITEMS_SLOTS_PATH = os.path.join(os.path.dirname(__file__), "..", "items_slots_configuration.csv")

SPLIT_QTY_THRESHOLD = 500     # single-order lines above this → split-order warehouse

# ── 1. Load raw data ───────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — LOAD RAW DATA")
print("=" * 65)

raw   = pd.read_csv(RAW_ORDER_PATH)
items = pd.read_csv(ITEMS_DICT_PATH)

print(f"  raw_order.csv        : {len(raw):>8,} rows  |  {raw['order_id'].nunique():,} unique orders"
      f"  |  {raw['item_code'].nunique():,} unique SKUs")
print(f"  items_dictionary.csv : {len(items):>8,} rows")

# ── 2. Remove duplicates & null values ────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — REMOVE DUPLICATES & NULL VALUES")
print("=" * 65)

raw_clean = raw.drop_duplicates().dropna()
raw_removed = len(raw) - len(raw_clean)
print(f"  raw_order  : {len(raw):,} → {len(raw_clean):,}  (removed {raw_removed:,} duplicate/null rows)")

items_clean = items.drop_duplicates().dropna(subset=["item_code", "box_volume"])
items_removed = len(items) - len(items_clean)
print(f"  items_dict : {len(items):,} → {len(items_clean):,}  (removed {items_removed:,} duplicate/null rows)")

# ── 3. Slot-fit filter — exclude SKUs that don't fit any pod slot type ────────
print("\n" + "=" * 65)
print("STEP 3 — SLOT-FIT FILTER  (items_slots_configuration.csv)")
print("=" * 65)

slots_cfg = pd.read_csv(ITEMS_SLOTS_PATH)
max_fit_per_sku = slots_cfg.groupby("item_code")["max_box_in_slot"].max()

oversized_sku_set = set(max_fit_per_sku[max_fit_per_sku <= 0].index)

# Items in items_dictionary but absent from slots config have no dimension conflict — treated as valid
valid_items     = items_clean[~items_clean["item_code"].isin(oversized_sku_set)].copy()
oversized_items = items_clean[ items_clean["item_code"].isin(oversized_sku_set)].copy()

valid_sku_set = set(valid_items["item_code"])

largest_slot_vol = int(slots_cfg["slot_volume"].max())
print(f"  Largest slot volume in system     : {largest_slot_vol:,} cm3")
print(f"  Valid SKUs (fit >= 1 slot type)   : {len(valid_sku_set):,}")
print(f"  Oversized SKUs -> big-item facility: {len(oversized_sku_set):,}")
if len(oversized_items) > 0:
    print(f"\n  Oversized SKU volume range:")
    print(f"    min : {oversized_items['box_volume'].min():,.0f} cm3")
    print(f"    max : {oversized_items['box_volume'].max():,.0f} cm3")
    print(f"    mean: {oversized_items['box_volume'].mean():,.0f} cm3")

# ── 4. Identify split-order SKUs ──────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 4 — SPLIT-ORDER FILTER  (max single-order qty > {SPLIT_QTY_THRESHOLD})")
print("=" * 65)

# Only consider valid-volume SKUs for this check
raw_vol_ok = raw_clean[raw_clean["item_code"].isin(valid_sku_set)].copy()

sku_max_qty = (
    raw_vol_ok.groupby("item_code")["item_quantity"]
    .max()
    .reset_index()
    .rename(columns={"item_quantity": "max_order_qty"})
)
split_sku_df    = sku_max_qty[sku_max_qty["max_order_qty"] > SPLIT_QTY_THRESHOLD].copy()
split_sku_set   = set(split_sku_df["item_code"])

# Attach item name and box volume for the report
split_sku_df = split_sku_df.merge(
    items_clean[["item_code", "item_name", "box_volume"]],
    on="item_code", how="left"
).sort_values("max_order_qty", ascending=False).reset_index(drop=True)

print(f"  Split-order SKUs (max qty > {SPLIT_QTY_THRESHOLD}) : {len(split_sku_set):,}")
print(f"\n  {'item_code':<16}  {'max_order_qty':>14}  {'box_volume':>12}  item_name")
print(f"  {'-'*70}")
for _, r in split_sku_df.iterrows():
    print(f"  {r['item_code']:<16}  {r['max_order_qty']:>14,.0f}  {r['box_volume']:>12,.0f}  {r['item_name']}")

# ── 5. Build final clean datasets ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — BUILD FINAL CLEAN DATASETS")
print("=" * 65)

clean_orders = raw_vol_ok[~raw_vol_ok["item_code"].isin(split_sku_set)].copy()
clean_skus   = valid_items[~valid_items["item_code"].isin(split_sku_set)].copy()

print(f"  Clean orders  : {len(clean_orders):,} rows"
      f"  |  {clean_orders['order_id'].nunique():,} orders"
      f"  |  {clean_orders['item_code'].nunique():,} unique SKUs")
print(f"  Clean SKU master : {len(clean_skus):,} SKUs")

# Routing summary
print(f"\n  Order routing summary:")
raw_oversized_rows = raw_clean[raw_clean["item_code"].isin(oversized_sku_set)]
raw_split_rows     = raw_vol_ok[raw_vol_ok["item_code"].isin(split_sku_set)]
print(f"    Main warehouse (clean)  : {len(clean_orders):,} order lines")
print(f"    Big-item facility        : {len(raw_oversized_rows):,} order lines"
      f"  ({len(oversized_sku_set):,} SKUs)")
print(f"    Split-order warehouse    : {len(raw_split_rows):,} order lines"
      f"  ({len(split_sku_set):,} SKUs)")
skus_not_in_dict = raw_clean[
    ~raw_clean["item_code"].isin(valid_sku_set) &
    ~raw_clean["item_code"].isin(oversized_sku_set)
]
print(f"    SKUs not in items_dict   : {len(skus_not_in_dict):,} order lines"
      f"  ({skus_not_in_dict['item_code'].nunique():,} SKUs)")

# ── 6. Save outputs ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — SAVE OUTPUTS")
print("=" * 65)

clean_orders.to_csv(os.path.join(OUTPUT_DIR, "00_clean_orders.csv"), index=False)
print(f"  Saved: output/00_clean_orders.csv  ({len(clean_orders):,} rows)")

clean_skus.to_csv(os.path.join(OUTPUT_DIR, "00_clean_skus.csv"), index=False)
print(f"  Saved: output/00_clean_skus.csv    ({len(clean_skus):,} rows)")

oversized_items.to_csv(os.path.join(OUTPUT_DIR, "00_oversized_skus.csv"), index=False)
print(f"  Saved: output/00_oversized_skus.csv  ({len(oversized_items):,} rows)")

split_sku_df.to_csv(os.path.join(OUTPUT_DIR, "00_split_order_skus.csv"), index=False)
print(f"  Saved: output/00_split_order_skus.csv  ({len(split_sku_df):,} rows)")

print("\nDone — Data cleaning complete.")
