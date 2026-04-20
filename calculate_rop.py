import pandas as pd
import numpy as np

items_dict = pd.read_csv('items_dictionary.csv')
items_csv = pd.read_csv('items.csv', index_col=0)
items_csv['item_id'] = items_csv.index
pods = pd.read_csv('pods.csv')

# Active SKUs only
active = items_dict[items_dict['slots_needed'] > 0].copy()
active = active.merge(items_csv[['item_code', 'item_id']], on='item_code', how='left')

# ROP global with lead_time = 1/8 day, Z = 1.28 (90% service level)
lead_time = 1 / 8
Z = 1.28
active['rop_global'] = (
    active['mean_daily_demand'] * lead_time +
    Z * active['std_daily_demand'] * np.sqrt(lead_time)
).clip(lower=1).round(0).astype(int)

# S_total: sum of max_qty across all pods per item
assigned = pods[pods['max_qty'] > 0]
s_total = assigned.groupby('item')['max_qty'].sum().reset_index()
s_total.columns = ['item_id', 's_total']

# Number of pods per item
pods_per_item = assigned.groupby('item')['pod_id'].nunique().reset_index()
pods_per_item.columns = ['item_id', 'num_pods']

# Merge
result = active.merge(s_total, on='item_id', how='left')
result = result.merge(pods_per_item, on='item_id', how='left')
result['s_total'] = result['s_total'].fillna(0).astype(int)
result['num_pods'] = result['num_pods'].fillna(1).astype(int)

# ROP per pod = ceil(rop_global / num_pods)
result['rop_per_pod'] = np.ceil(result['rop_global'] / result['num_pods']).astype(int)

# Final output
output = result[['item_code', 'item_id', 'num_pods', 's_total', 'rop_global', 'rop_per_pod']].copy()
output = output.rename(columns={'num_pods': 'n_slots'})
output = output.sort_values('item_id').reset_index(drop=True)

output.to_csv('rop_summary.csv', index=False)
print(f"Saved rop_summary.csv — {len(output)} SKUs")
print(f"\nLead time: 1/8 day = 3 hours, Z = 1.28 (90% service level)")
print(f"\nROP global stats:")
print(output['rop_global'].describe())
print(f"\nROP per pod stats:")
print(output['rop_per_pod'].describe())
print(f"\nS total stats:")
print(output['s_total'].describe())
print(f"\nSample output:")
print(output.head(10).to_string())
