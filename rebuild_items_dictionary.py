import pandas as pd
import numpy as np
from scipy.stats import norm

CLEAN_SKUS_PATH   = 'data_mining/output/00_clean_skus.csv'
CLEAN_ORDERS_PATH = 'data_mining/output/00_clean_orders.csv'
CV_CLASS_PATH     = 'data_mining/output/00e_cv_classification.csv'
SLOTS_PATH        = 'items_slots_configuration.csv'

# Z score per (ABC class, CV class) — Table 4.5
Z_MAP = {
    ('A', 0): norm.ppf(0.99),   # 2.3263 — high demand stable
    ('A', 1): norm.ppf(0.95),   # 1.6449 — high demand volatile
    ('A', 2): norm.ppf(0.90),   # 1.2816 — high demand intermittent
    ('B', 1): norm.ppf(0.95),   # 1.6449 — medium demand volatile
    ('B', 2): norm.ppf(0.90),   # 1.2816 — medium demand intermittent
    ('C', 2): norm.ppf(0.90),   # 1.2816 — low demand intermittent
}

# ── 1. Load base SKU list (7,431 clean SKUs) ──────────────────────────────────
df       = pd.read_csv(CLEAN_SKUS_PATH)
cv_df    = pd.read_csv(CV_CLASS_PATH)[['item_code', 'abc_kmeans', 'cv', 'cv_class']]
orders   = pd.read_csv(CLEAN_ORDERS_PATH)
slots    = pd.read_csv(SLOTS_PATH)

print(f'Base SKUs (clean_skus.csv)  : {len(df):,}')
print(f'CV classification SKUs      : {len(cv_df):,}')

# ── 2. Daily demand stats from clean orders ───────────────────────────────────
orders['order_date'] = pd.to_datetime(orders['order_date'], format='%m/%d/%Y %H:%M')
orders['date']       = orders['order_date'].dt.date
all_dates            = sorted(orders['date'].unique())
total_days           = len(all_dates)
print(f'Clean orders days           : {total_days}  ({all_dates[0]} to {all_dates[-1]})')

sku_list = df['item_code'].unique()
daily    = orders.groupby(['item_code', 'date'])['item_quantity'].sum().reset_index()
full_idx = pd.MultiIndex.from_product([sku_list, all_dates], names=['item_code', 'date'])
daily_full = (
    daily.set_index(['item_code', 'date'])
    .reindex(full_idx, fill_value=0)
    .reset_index()
)
demand_stats = daily_full.groupby('item_code')['item_quantity'].agg(['mean', 'std']).reset_index()
demand_stats.columns = ['item_code', 'mean_daily_demand', 'std_daily_demand']
demand_stats['std_daily_demand']  = demand_stats['std_daily_demand'].fillna(0)
demand_stats['mean_daily_demand'] = demand_stats['mean_daily_demand'].round(4)
demand_stats['std_daily_demand']  = demand_stats['std_daily_demand'].round(4)

# ── 3. Merge ABC class and CV from data mining outputs ────────────────────────
stats = demand_stats.merge(cv_df, on='item_code', how='left')
stats['item_class'] = stats['abc_kmeans']
# cv column: use CV score from 00e (based on order frequency distribution)
# already present as 'cv' from cv_df merge

# ── 4. Z score per (ABC class, CV class) ─────────────────────────────────────
stats['Z'] = stats.apply(
    lambda r: Z_MAP.get((r['item_class'], r['cv_class']), norm.ppf(0.90)), axis=1
)

# ── 5. Item initial quantity inventory — lead_time = 1 day ───────────────────
lead_time_init = 1.0
stats['item_initial_quantity_inventory'] = (
    stats['mean_daily_demand'] * lead_time_init
    + stats['Z'] * stats['std_daily_demand'] * np.sqrt(lead_time_init)
).clip(lower=1).apply(np.ceil).astype(int)

# ── 6. ROP global — lead_time = 1/8 day (one shift) ─────────────────────────
lead_time_rop = 1 / 8
stats['rop_global'] = (
    stats['mean_daily_demand'] * lead_time_rop
    + stats['Z'] * stats['std_daily_demand'] * np.sqrt(lead_time_rop)
).clip(lower=1).round(0).astype(int)

# ── 7. Slots needed — all 7,431 SKUs, slot_type=2 (60,000 cm³) ───────────────
slots2 = (
    slots[slots['slot_type'] == 2]
    .drop_duplicates('item_code')[['item_code', 'max_box_in_slot']]
)
stats = stats.merge(slots2, on='item_code', how='left')
stats = stats.merge(df[['item_code', 'number_of_item_in_a_box', 'box_volume']], on='item_code', how='left')

stats['number_of_item_in_a_box'] = stats['number_of_item_in_a_box'].replace(0, 1)
# For any SKU missing from slots config, estimate from box_volume
stats['max_box_in_slot'] = stats['max_box_in_slot'].fillna(
    (60000 / stats['box_volume']).astype(int).clip(lower=1)
)
# Truly oversized (max_box_in_slot == 0 across all slot types) → slots_needed = 0
slots_max = slots.groupby('item_code')['max_box_in_slot'].max()
oversized_set = set(slots_max[slots_max <= 0].index)
stats['slots_needed'] = np.ceil(
    stats['item_initial_quantity_inventory']
    / stats['number_of_item_in_a_box']
    / stats['max_box_in_slot'].clip(lower=1)
).replace([np.inf, -np.inf], 1).fillna(1).clip(lower=1).astype(int)
stats.loc[stats['item_code'].isin(oversized_set), 'slots_needed'] = 0

# ── 8. Merge computed columns back into base SKU master ───────────────────────
computed_cols = [
    'item_code', 'item_class', 'item_initial_quantity_inventory',
    'mean_daily_demand', 'std_daily_demand', 'cv', 'rop_global', 'slots_needed'
]
drop_cols = [c for c in computed_cols[1:] if c in df.columns]
df = df.drop(columns=drop_cols)
df = df.merge(stats[computed_cols], on='item_code', how='left')

# ── 9. Final column order ─────────────────────────────────────────────────────
final_cols = [
    'item_code', 'item_name', 'box_length', 'box_width', 'box_height',
    'box_volume', 'box_weight', 'number_of_item_in_a_box', 'item_volume',
    'item_unit', 'item_order_frequency', 'item_relative_importance',
    'item_quantity_order_mean', 'item_quantity_order_std',
    'item_quantity_order_total', 'item_quantity_order_unique',
    'item_quantity_order_unique_frequency', 'item_class',
    'item_initial_quantity_inventory', 'mean_daily_demand', 'std_daily_demand',
    'cv', 'rop_global', 'slots_needed',
]
df = df[final_cols]
df.to_csv('items_dictionary.csv', index=False)

# ── 10. Summary ───────────────────────────────────────────────────────────────
print(f'\nitems_dictionary.csv rebuilt: {len(df):,} rows, {len(df.columns)} columns')
print(f'\nABC distribution (K-Means):')
print(df['item_class'].value_counts().sort_index().to_string())

total_slots = int(df['slots_needed'].sum())
pods_needed = int(np.ceil(total_slots / 20))
print(f'\nTotal slots needed : {total_slots:,}')
print(f'Pods needed (20 slots/pod): {pods_needed}')

print(f'\nInitial inventory stats:')
print(df['item_initial_quantity_inventory'].describe().round(1).to_string())

print(f'\nService level by class (Z scores used):')
for (abc, cv_cls), z in sorted(Z_MAP.items()):
    cnt = len(stats[(stats['item_class'] == abc) & (stats['cv_class'] == cv_cls)])
    sl  = norm.cdf(z) * 100
    print(f'  {abc} x CV{cv_cls}: Z={z:.4f} ({sl:.0f}% SL)  —  {cnt:,} SKUs')

print(f'\n>>> Update netlogo.py: total_sku={len(df):,}, pods_needed={pods_needed}')
