import pandas as pd
import numpy as np

items_dict = pd.read_csv('items_dictionary.csv')
items_csv = pd.read_csv('items.csv', index_col=0)
items_csv['item_id'] = items_csv.index
pods = pd.read_csv('pods.csv')

# Active SKUs only — inner join keeps only the selected SKUs in items.csv
active = items_dict[items_dict['slots_needed'] > 0].copy()
active = active.merge(items_csv[['item_code', 'item_id']], on='item_code', how='inner')

# rop_global already computed per-SKU with correct per-class Z in items_dictionary.csv
active['rop_global'] = active['rop_global'].clip(lower=1).round(0).astype(int)

# S_total: sum of max_qty across all pods per item
assigned = pods[pods['max_qty'] > 0]
s_total = assigned.groupby('item')['max_qty'].sum().reset_index()
s_total.columns = ['item_id', 's_total']

# Merge
result = active.merge(s_total, on='item_id', how='left')
result['s_total'] = result['s_total'].fillna(0).astype(int)

# n_slots = total slots that hold this SKU across the warehouse (items_dictionary).
# ROP per slot = ceil(rop_global / n_slots). The per-slot reorder point makes the
# pod-level trigger proportional: a pod's slot is "low" iff its per-slot stock
# (current_qty / n_slots_in_pod) <= rop_per_slot, regardless of how many slots a
# given pod devotes to the SKU (pods may hold the same SKU in different slot counts).
result['n_slots'] = result['slots_needed'].clip(lower=1).astype(int)
result['rop_per_slot'] = np.ceil(result['rop_global'] / result['n_slots']).astype(int)

# Final output
output = result[['item_code', 'item_id', 'n_slots', 's_total', 'rop_global', 'rop_per_slot']].copy()
output = output.sort_values('item_id').reset_index(drop=True)

output.to_csv('rop_summary.csv', index=False)
print(f"Saved rop_summary.csv — {len(output)} SKUs")
print(f"\nROP global read from items_dictionary.csv (per-class Z: A/CV0=99%, A/CV1=B/CV1=95%, rest=90%)")
print(f"\nROP global stats:")
print(output['rop_global'].describe())
print(f"\nROP per slot stats:")
print(output['rop_per_slot'].describe())
print(f"\nS total stats:")
print(output['s_total'].describe())
print(f"\nSample output:")
print(output.head(10).to_string())
