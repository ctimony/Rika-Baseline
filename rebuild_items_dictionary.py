import pandas as pd
import numpy as np

raw = pd.read_csv('raw_order.csv')
df = pd.read_csv('items_dictionary.csv')
slots = pd.read_csv('items_slots_configuration.csv')

raw['order_date'] = pd.to_datetime(raw['order_date'])
raw['date'] = raw['order_date'].dt.date
total_days = raw['date'].nunique()  # 21
all_dates = raw['date'].unique()

# --- Mean & Std Daily Demand (include zero-demand days) ---
daily = raw.groupby(['item_code', 'date'])['item_quantity'].sum().reset_index()
full_idx = pd.MultiIndex.from_product(
    [df['item_code'].unique(), all_dates], names=['item_code', 'date'])
daily_full = daily.set_index(['item_code', 'date']).reindex(full_idx, fill_value=0).reset_index()

stats = daily_full.groupby('item_code')['item_quantity'].agg(['mean', 'std']).reset_index()
stats.columns = ['item_code', 'mean_daily_demand', 'std_daily_demand']
stats['std_daily_demand'] = stats['std_daily_demand'].fillna(0)
stats['cv'] = (stats['std_daily_demand'] / stats['mean_daily_demand'].replace(0, np.nan)).fillna(0).round(4)
stats['mean_daily_demand'] = stats['mean_daily_demand'].round(4)
stats['std_daily_demand'] = stats['std_daily_demand'].round(4)

# --- ABC Classification (cumulative daily demand Pareto) ---
stats = stats.sort_values('mean_daily_demand', ascending=False).copy()
stats['cum_pct'] = stats['mean_daily_demand'].cumsum() / stats['mean_daily_demand'].sum()
stats['abc'] = 'C'
stats.loc[stats['cum_pct'] <= 0.70, 'abc'] = 'A'
stats.loc[(stats['cum_pct'] > 0.70) & (stats['cum_pct'] <= 0.90), 'abc'] = 'B'

# --- XYZ Classification (by CV) ---
stats['xyz'] = 'Z'
stats.loc[stats['cv'] <= 0.5, 'xyz'] = 'X'
stats.loc[(stats['cv'] > 0.5) & (stats['cv'] <= 1.0), 'xyz'] = 'Y'

stats['abc_xyz'] = stats['abc'] + stats['xyz']
stats['item_class'] = stats['abc']

# --- ROP Global (integer) ---
lead_time_days = 1 / 24  # 1/8 shift / 3 shifts per day
Z = 1.28  # 90% service level
stats['rop_global'] = (
    stats['mean_daily_demand'] * lead_time_days +
    Z * stats['std_daily_demand'] * np.sqrt(lead_time_days)
).clip(lower=1).round(0).astype(int)

# --- Item Initial Quantity Inventory: ROP(lead_time=1 day, k=1) ---
# Initial stock = 1-day safety buffer; ROP trigger remains at lead_time=1/8 day
stats['item_initial_quantity_inventory'] = (
    stats['mean_daily_demand'] * 1 + Z * stats['std_daily_demand'] * np.sqrt(1)
).clip(lower=1).apply(np.ceil).astype(int)

# --- Slots Needed (pod_type=3, slot_type=5, vol=60,000) ---
stats = stats.merge(df[['item_code', 'box_volume', 'number_of_item_in_a_box']], on='item_code', how='left')
excluded = stats[stats['box_volume'] >= 60000].copy()
fitting_all = stats[stats['box_volume'] < 60000].copy()
# Take top 4,000 SKUs by mean_daily_demand (all A+B preserved, top C by demand)
fitting = fitting_all.sort_values('mean_daily_demand', ascending=False).head(2000).copy()
excluded_low_demand = fitting_all[~fitting_all['item_code'].isin(fitting['item_code'])].copy()
excluded_low_demand['slots_needed'] = 0
print(f'Excluded {len(excluded)} SKUs (box_volume >= 60,000, do not fit pod_type=3)')
print(f'Excluded {len(excluded_low_demand)} C-class SKUs (low demand, beyond top 4,000)')
print(f'Active SKUs: {len(fitting)}')

slots5 = slots[slots['slot_type'] == 5].drop_duplicates('item_code')[['item_code', 'max_box_in_slot']]
fitting = fitting.merge(slots5, on='item_code', how='left')
fitting['max_box_in_slot'] = fitting['max_box_in_slot'].fillna(
    (60000 / fitting['box_volume']).astype(int).clip(lower=1))
fitting['number_of_item_in_a_box'] = fitting['number_of_item_in_a_box'].replace(0, 1)
fitting['slots_needed'] = np.ceil(
    fitting['item_initial_quantity_inventory'] /
    fitting['number_of_item_in_a_box'] /
    fitting['max_box_in_slot']
).replace([np.inf, -np.inf], 1).fillna(1).clip(lower=1).astype(int)

excluded['slots_needed'] = 0

total_slots = int(fitting['slots_needed'].sum())
pods_needed = int(np.ceil(total_slots / 20))
print(f'\nTotal slots needed: {total_slots:,}')
print(f'Pods needed (pod_type=3, 20 slots/pod): {pods_needed}')
print(f'\nABC distribution:')
print(fitting['item_class'].value_counts())
print(f'\nABC x XYZ distribution:')
print(fitting['abc_xyz'].value_counts().sort_index())

# --- Merge back into items_dictionary (all 7,442 rows) ---
# slots_needed=0 for excluded (vol too large) and low-demand C-class not in active set
all_stats = pd.concat([
    fitting[['item_code', 'item_class', 'item_initial_quantity_inventory',
             'mean_daily_demand', 'std_daily_demand', 'cv', 'abc_xyz', 'rop_global', 'slots_needed']],
    excluded[['item_code', 'item_class', 'item_initial_quantity_inventory',
              'mean_daily_demand', 'std_daily_demand', 'cv', 'abc_xyz', 'rop_global', 'slots_needed']],
    excluded_low_demand[['item_code', 'item_class', 'item_initial_quantity_inventory',
                         'mean_daily_demand', 'std_daily_demand', 'cv', 'abc_xyz', 'rop_global', 'slots_needed']],
], ignore_index=True)

drop_cols = ['item_class', 'item_initial_quantity_inventory',
             'mean_daily_demand', 'std_daily_demand', 'cv', 'abc_xyz', 'rop_global', 'slots_needed']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
df = df.merge(all_stats, on='item_code', how='left')

cols = [
    'item_code', 'item_name', 'box_length', 'box_width', 'box_height',
    'box_volume', 'box_weight', 'number_of_item_in_a_box', 'item_volume',
    'item_unit', 'item_order_frequency', 'item_relative_importance',
    'item_quantity_order_mean', 'item_quantity_order_std',
    'item_quantity_order_total', 'item_quantity_order_unique',
    'item_quantity_order_unique_frequency', 'item_class',
    'item_initial_quantity_inventory',
    'mean_daily_demand', 'std_daily_demand', 'cv', 'abc_xyz', 'rop_global', 'slots_needed'
]
df = df[cols]
df.to_csv('items_dictionary.csv', index=False)
print(f'\nitems_dictionary.csv rebuilt: {len(df)} rows, {len(df.columns)} columns')
active_sku = len(fitting)
print(f'\n>>> Update netlogo.py: pod_types=[3], pod_num=[{pods_needed}], total_sku={active_sku}')
print(f'>>> Also update total_requested_item={active_sku} (lines 267 and 283)')
