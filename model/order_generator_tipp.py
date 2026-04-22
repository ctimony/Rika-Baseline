import os
import numpy as np
import pandas as pd


def gen_order_tipp(
    raw_order_path: str = 'raw_order.csv',
    sim_duration_sec: int = 28800,
    start_hour: int = 16,
    quantity_range: list = None,
    date: int = 1,
    initial_backlog: int = 100,
):
    """Generate orders using a Time-Inhomogeneous Poisson Process (TIPP).

    Uses 24 empirical hourly rates from raw_order.csv directly — no H/L phase
    classification. Outputs generated_order.csv in the same 6-column format used
    by the simulation.
    """
    if quantity_range is None:
        quantity_range = [1, 12]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_dir, 'generated_order.csv')

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        existing_duration = int(existing['order_arrival'].max())
        if existing_duration >= sim_duration_sec - 1:
            print("    generated_order.csv already exists — skipping TIPP generation.")
            return
        print("    generated_order.csv exists but duration mismatch — regenerating.")

    print("TIPP: generating orders...")

    # ── Hourly arrival rates ───────────────────────────────────────────────────
    raw = pd.read_csv(os.path.join(base_dir, raw_order_path))
    raw['order_date'] = pd.to_datetime(raw['order_date'], format='%m/%d/%Y %H:%M')
    raw['hour'] = raw['order_date'].dt.hour
    raw['date_only'] = raw['order_date'].dt.date

    hourly_avg = (
        raw.groupby(['date_only', 'hour'])['order_id']
        .nunique()
        .groupby('hour')
        .mean()
    )
    # orders/sec for each clock hour
    rate_by_hour = {h: float(hourly_avg.get(h, hourly_avg.mean())) / 3600.0
                    for h in range(24)}

    # ── SKU probabilities ──────────────────────────────────────────────────────
    items_path = os.path.join(base_dir, 'items.csv')
    items_active = pd.read_csv(items_path)[['item_code', 'item_id']]

    sku_freq = (
        raw.groupby('item_code')['order_id']
        .count()
        .reset_index()
        .rename(columns={'order_id': 'freq'})
    )
    sku_freq = sku_freq.merge(items_active, on='item_code', how='inner')
    sku_freq['prob'] = sku_freq['freq'] / sku_freq['freq'].sum()
    sku_ids = sku_freq['item_id'].astype(int).tolist()
    sku_probs = sku_freq['prob'].tolist()

    # ── Quantity distribution per SKU ──────────────────────────────────────────
    qty_stats = (
        raw.assign(item_quantity=raw['item_quantity'].clip(upper=quantity_range[1]))
        .groupby('item_code')['item_quantity']
        .agg(['mean', 'std'])
        .reset_index()
    )
    qty_stats['std'] = qty_stats['std'].fillna(1.0).clip(lower=0.5)
    qty_stats = qty_stats.merge(items_active, on='item_code', how='inner')
    qty_mean_by_id = dict(zip(qty_stats['item_id'].astype(int), qty_stats['mean']))
    qty_std_by_id  = dict(zip(qty_stats['item_id'].astype(int), qty_stats['std']))

    global_qty_mean = qty_stats['mean'].mean()
    global_qty_std  = qty_stats['std'].mean()

    # ── Items per order (Negative Binomial fitted from raw_order.csv) ─────────
    # r=0.6957, p=0.1148 → mean=5.36, std=6.91 items/order
    NBINOM_R = 0.6957
    NBINOM_P = 0.1148

    def sample_n_items():
        return max(1, int(np.random.negative_binomial(NBINOM_R, NBINOM_P)))

    def sample_qty(item_id):
        mu  = qty_mean_by_id.get(item_id, global_qty_mean)
        sig = qty_std_by_id.get(item_id, global_qty_std)
        q = int(round(np.random.normal(mu, sig)))
        return max(quantity_range[0], min(quantity_range[1], q))

    # ── Generate backlog orders (order_arrival = 0, order_id negative) ─────────
    rows = []
    for i in range(1, initial_backlog + 1):
        order_id = -i
        n_items = sample_n_items()
        chosen = np.random.choice(len(sku_ids), size=min(n_items, len(sku_ids)),
                                  replace=False, p=sku_probs)
        for idx in chosen:
            item_id = sku_ids[idx]
            rows.append({
                'order_id':      order_id,
                'order_type':    1,
                'item_id':       item_id,
                'item_quantity': sample_qty(item_id),
                'order_arrival': 0,
            })

    # ── Generate shift orders via inter-arrival sampling ───────────────────────
    t_next = 0.0
    order_id = 0
    while t_next < sim_duration_sec:
        clock_hour = (start_hour + int(t_next) // 3600) % 24
        lam = rate_by_hour[clock_hour]
        inter = np.random.exponential(1.0 / lam)
        t_next += inter
        if t_next >= sim_duration_sec:
            break

        arrival_sec = int(t_next)
        n_items = sample_n_items()
        chosen = np.random.choice(len(sku_ids), size=min(n_items, len(sku_ids)),
                                  replace=False, p=sku_probs)
        for idx in chosen:
            item_id = sku_ids[idx]
            rows.append({
                'order_id':      order_id,
                'order_type':    1,
                'item_id':       item_id,
                'item_quantity': sample_qty(item_id),
                'order_arrival': arrival_sec,
            })
        order_id += 1

    # ── Write output ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=['order_id', 'order_type', 'item_id',
                                     'item_quantity', 'order_arrival'])
    df = df.sort_values(['order_arrival', 'order_id']).reset_index(drop=True)
    df.insert(0, 'sequence_id', df.index)

    df.to_csv(out_path, index=False)

    n_backlog = df[df['order_id'] < 0]['order_id'].nunique()
    n_shift   = df[df['order_id'] >= 0]['order_id'].nunique()
    avg_rate  = n_shift / (sim_duration_sec / 3600)
    print(f"TIPP: {n_backlog} backlog orders + {n_shift} shift orders "
          f"({avg_rate:.1f} orders/hr avg) → generated_order.csv")
