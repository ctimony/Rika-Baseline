"""
00b_data_filtering.py — SKU Demand Feature Engineering (post-cleaning)

Reads the cleaned order and SKU datasets produced by 00_data_cleaning.py and
computes per-SKU demand statistics that will feed the K-Means clustering step.

Features computed per SKU:
  - order_frequency   : number of unique orders containing the SKU
  - total_qty         : total units ordered across all orders
  - avg_qty_per_order : mean order line quantity
  - std_qty           : standard deviation of order line quantity
  - demand_cv         : coefficient of variation (std / mean)
  - days_ordered      : number of distinct days on which the SKU was ordered
  - avg_basket_size   : average number of SKUs per order that this SKU appears in

Outputs (data_mining/output/):
  00b_sku_demand_features.csv — one row per SKU with all demand features

Prerequisite: run 00_data_cleaning.py first.
Run: python data_mining/00b_data_filtering.py
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLEAN_ORDERS_PATH = os.path.join(OUTPUT_DIR, "00_clean_orders.csv")
CLEAN_SKUS_PATH   = os.path.join(OUTPUT_DIR, "00_clean_skus.csv")

# ── 1. Load cleaned data ───────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — LOAD CLEANED DATA")
print("=" * 65)

orders = pd.read_csv(CLEAN_ORDERS_PATH)
skus   = pd.read_csv(CLEAN_SKUS_PATH)

print(f"  Clean orders : {len(orders):,} rows  |  {orders['order_id'].nunique():,} unique orders"
      f"  |  {orders['item_code'].nunique():,} unique SKUs")
print(f"  Clean SKUs   : {len(skus):,} rows")

# ── 2. Parse order_date ────────────────────────────────────────────────────────
orders["order_date"] = pd.to_datetime(orders["order_date"], format="%m/%d/%Y %H:%M")
orders["date"]       = orders["order_date"].dt.date

date_min = orders["order_date"].min()
date_max = orders["order_date"].max()
n_days   = orders["date"].nunique()
print(f"\n  Date range : {date_min.date()} → {date_max.date()}  ({n_days} distinct days)")

# ── 3. Basket size lookup (orders → number of distinct SKUs) ──────────────────
basket_size = (
    orders.groupby("order_id")["item_code"]
    .nunique()
    .reset_index()
    .rename(columns={"item_code": "basket_size"})
)
orders = orders.merge(basket_size, on="order_id", how="left")

# ── 4. Compute per-SKU demand features ────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — COMPUTE PER-SKU DEMAND FEATURES")
print("=" * 65)

sku_features = (
    orders.groupby("item_code")
    .agg(
        order_frequency  = ("order_id",      "nunique"),
        total_qty        = ("item_quantity",  "sum"),
        avg_qty_per_order= ("item_quantity",  "mean"),
        std_qty          = ("item_quantity",  "std"),
        days_ordered     = ("date",           "nunique"),
        avg_basket_size  = ("basket_size",    "mean"),
    )
    .reset_index()
)

sku_features["std_qty"]   = sku_features["std_qty"].fillna(0.0)
sku_features["demand_cv"] = sku_features.apply(
    lambda r: r["std_qty"] / r["avg_qty_per_order"] if r["avg_qty_per_order"] > 0 else 0.0,
    axis=1
)

# Attach item name from orders
name_map = (
    orders.groupby("item_code")["item_name"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)
sku_features = sku_features.merge(name_map, on="item_code", how="left")

# Attach box dimensions from clean SKU master
dim_cols = ["item_code", "box_length", "box_width", "box_height", "box_volume", "item_volume"]
sku_features = sku_features.merge(skus[dim_cols], on="item_code", how="left")

print(f"  SKUs with demand features : {len(sku_features):,}")
print(f"\n  order_frequency  — min: {sku_features['order_frequency'].min()}"
      f"  |  max: {sku_features['order_frequency'].max()}"
      f"  |  mean: {sku_features['order_frequency'].mean():.1f}")
print(f"  total_qty        — min: {sku_features['total_qty'].min()}"
      f"  |  max: {sku_features['total_qty'].max()}"
      f"  |  mean: {sku_features['total_qty'].mean():.1f}")
print(f"  demand_cv        — min: {sku_features['demand_cv'].min():.3f}"
      f"  |  max: {sku_features['demand_cv'].max():.3f}"
      f"  |  mean: {sku_features['demand_cv'].mean():.3f}")
print(f"  box_volume       — min: {sku_features['box_volume'].min():,.0f}"
      f"  |  max: {sku_features['box_volume'].max():,.0f}"
      f"  |  mean: {sku_features['box_volume'].mean():,.0f}")

# ── 5. SKUs in clean master but with zero orders ───────────────────────────────
ordered_skus  = set(sku_features["item_code"])
all_clean_skus = set(skus["item_code"])
no_order_skus  = all_clean_skus - ordered_skus
print(f"\n  SKUs in clean master with no orders in 21-day window : {len(no_order_skus):,}")

# ── 6. Distribution snapshot ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — FREQUENCY DISTRIBUTION SNAPSHOT")
print("=" * 65)
bins  = [0, 1, 5, 10, 25, 50, 100, 250, 500, float("inf")]
labels= ["1","2-5","6-10","11-25","26-50","51-100","101-250","251-500","500+"]
sku_features["freq_band"] = pd.cut(
    sku_features["order_frequency"], bins=bins, labels=labels, right=True
)
freq_dist = sku_features["freq_band"].value_counts().reindex(labels)
print(f"  {'Band':<12}  {'SKUs':>8}  {'%':>7}")
print(f"  {'-'*30}")
for band, cnt in freq_dist.items():
    pct = cnt / len(sku_features) * 100
    print(f"  {band:<12}  {cnt:>8,}  {pct:>6.1f}%")
sku_features.drop(columns=["freq_band"], inplace=True)

# ── 7. Save ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — SAVE OUTPUTS")
print("=" * 65)

col_order = [
    "item_code", "item_name",
    "order_frequency", "total_qty", "avg_qty_per_order",
    "std_qty", "demand_cv", "days_ordered", "avg_basket_size",
    "box_length", "box_width", "box_height", "box_volume", "item_volume",
]
sku_features[col_order].to_csv(
    os.path.join(OUTPUT_DIR, "00b_sku_demand_features.csv"), index=False
)
print(f"  Saved: output/00b_sku_demand_features.csv  ({len(sku_features):,} rows)")

print("\nDone — SKU demand feature engineering complete.")
