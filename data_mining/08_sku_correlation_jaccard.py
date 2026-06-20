"""SKU correlation analysis using Jaccard similarity.

Mengukur korelasi antar SKU dengan Jaccard similarity, mengikuti
Ma et al. (2023), "A Novel Scattered Storage Policy Considering Commodity
Classification and Correlation in RMFS", IEEE TASE. Mereka mendefinisikan
"correlation degree" antara SKU i dan j (Eq. 13) sebagai:

    s_ij = n(i ∩ j) / ( n(i) + n(j) - n(i ∩ j) )

dengan
    n(i ∩ j) = jumlah order yang memuat KEDUA SKU i dan j,
    n(i), n(j) = jumlah order yang memuat masing-masing SKU.

Ini identik dengan Jaccard similarity antar himpunan order tiap SKU.

Tujuan analisis: menguji apakah korelasi antar SKU pada data order riil
cukup kuat untuk dijadikan dasar kebijakan penyimpanan (scattered storage).
"""

import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORDERS_CSV = os.path.join(HERE, "output", "02_clean_orders.csv")
OUT_DIR = os.path.join(HERE, "output")

# Minimum order-count per SKU to be considered (drop ultra-rare SKUs that
# only create spurious high-similarity pairs from a handful of co-occurrences).
MIN_ORDERS_PER_SKU = 50
# Minimum co-occurrence count for a pair to be reported.
MIN_PAIR_COOC = 20


def load_baskets(path):
    """Return list of order-baskets (sets of SKU codes) and a Series of
    per-SKU order counts."""
    df = pd.read_csv(path, dtype={"order_id": str, "item_code": str})
    # one (order, sku) presence regardless of quantity / duplicate lines
    df = df[["order_id", "item_code"]].drop_duplicates()
    baskets = df.groupby("order_id")["item_code"].apply(set)
    sku_orders = df.groupby("item_code")["order_id"].nunique()
    return baskets, sku_orders


def prevalence_stats(baskets):
    """Order-structure statistics that characterize how much correlation there
    even is to exploit (cf. Zhuang et al. 2024). Returns a dict."""
    sizes = baskets.apply(len)
    n_orders = len(baskets)
    single = (sizes == 1).sum()
    return {
        "n_orders": n_orders,
        "mean_size": sizes.mean(),
        "median_size": sizes.median(),
        "pct_single": 100 * single / n_orders,
        "pct_multi": 100 * (n_orders - single) / n_orders,
    }


def compute_jaccard(baskets, sku_orders):
    """Compute Jaccard correlation degree s_ij (Ma et al. Eq. 13) for every
    qualifying SKU pair. Returns a DataFrame."""
    n_orders = baskets.shape[0]
    eligible = set(sku_orders[sku_orders >= MIN_ORDERS_PER_SKU].index)

    # co-occurrence counts over eligible SKUs only
    cooc = {}
    for basket in baskets:
        items = sorted(basket & eligible)
        for a, c in combinations(items, 2):
            cooc[(a, c)] = cooc.get((a, c), 0) + 1

    rows = []
    for (a, c), n_ac in cooc.items():
        if n_ac < MIN_PAIR_COOC:
            continue
        n_a = sku_orders[a]
        n_c = sku_orders[c]
        union = n_a + n_c - n_ac
        s_ij = n_ac / union if union > 0 else 0.0
        rows.append((a, c, n_ac, int(n_a), int(n_c), s_ij))

    out = pd.DataFrame(
        rows, columns=["sku_i", "sku_j", "cooc", "n_i", "n_j", "jaccard"]
    ).sort_values("jaccard", ascending=False, ignore_index=True)
    return out, n_orders, len(eligible)


# Order-size break-even below which turnover-based (ABC) storage beats
# correlation-based (CDA/CRL) storage, per Mirzaei, Zaerpour & de Koster
# (2022), IJPR, Observation 8 (~3.8-4 lines/order).
MIRZAEI_BREAKEVEN = 3.8


def main():
    baskets, sku_orders = load_baskets(ORDERS_CSV)
    pairs, n_orders, n_eligible = compute_jaccard(baskets, sku_orders)
    prev = prevalence_stats(baskets)

    jac = pairs["jaccard"].values
    median = np.median(jac)
    p90 = np.percentile(jac, 90)
    mx = jac.max()
    print(f"Orders            : {n_orders}")
    print(f"Eligible SKUs     : {n_eligible} (>= {MIN_ORDERS_PER_SKU} orders)")
    print(f"Pairs (>= {MIN_PAIR_COOC} cooc): {len(pairs)}")
    print(f"Jaccard median    : {median:.4f}")
    print(f"Jaccard 90th pct  : {p90:.4f}")
    print(f"Jaccard max       : {mx:.4f}")
    print(f"Pairs >= 0.10     : {(jac >= 0.10).sum()} "
          f"({100*(jac >= 0.10).mean():.1f}%)")
    print(f"Pairs >= 0.20     : {(jac >= 0.20).sum()} "
          f"({100*(jac >= 0.20).mean():.1f}%)")
    print("-- order structure --")
    print(f"Mean SKU/order    : {prev['mean_size']:.2f}")
    print(f"Median SKU/order  : {prev['median_size']:.0f}")
    print(f"Single-SKU orders : {prev['pct_single']:.1f}%")
    print(f"Multi-SKU orders  : {prev['pct_multi']:.1f}%")

    pairs.to_csv(os.path.join(OUT_DIR, "08_sku_correlation_jaccard.csv"),
                 index=False)

    # ---- chart: Jaccard-only, two panels ----------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle(
        "SKU Correlation — Jaccard Similarity (Ma et al. 2023, Eq. 13)\n"
        f"{n_orders:,} orders, {n_eligible} eligible SKUs, "
        f"{len(pairs):,} pairs (min {MIN_ORDERS_PER_SKU}/SKU, "
        f"{MIN_PAIR_COOC}/pair)",
        fontweight="bold",
    )

    # Panel 1: distribution of Jaccard correlation degree
    ax1.hist(jac, bins=40, color="#4c72b0", edgecolor="white", linewidth=0.4)
    ax1.axvline(median, color="red", linestyle="--", linewidth=1.6,
                label=f"median = {median:.3f}")
    ax1.axvline(p90, color="#dd8452", linestyle="--", linewidth=1.4,
                label=f"90th pct = {p90:.3f}")
    ax1.set_xlabel("Jaccard similarity  $s_{ij}$")
    ax1.set_ylabel("# SKU pairs")
    ax1.set_title("Distribution of correlation degree")
    ax1.legend()

    # Panel 2: CDF — almost all pairs are very low (no arbitrary cutoff)
    xs = np.sort(jac)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax2.plot(xs, ys, color="#4c72b0", linewidth=1.8)
    ax2.axvline(median, color="red", linestyle="--", linewidth=1.4)
    frac_below_med = (jac < median).mean()
    ax2.annotate(f"50% of pairs\n< {median:.3f}",
                 xy=(median, frac_below_med), xytext=(median + 0.07, 0.45),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 color="red")
    ax2.set_xlabel("Jaccard similarity  $s_{ij}$")
    ax2.set_ylabel("Cumulative share of pairs")
    ax2.set_title("Correlation is weak: nearly all pairs are low")
    ax2.set_ylim(0, 1.02)

    # Footer: tie weak correlation + order size to Mirzaei et al. (2022)
    # break-even — the decision rule for whether correlation storage helps.
    footer = (
        f"Mean order size = {prev['mean_size']:.2f} lines  "
        f"(Mirzaei et al. 2022 break-even ≈ {MIRZAEI_BREAKEVEN} lines/order "
        f"for correlation-based storage to beat ABC);  "
        f"{prev['pct_single']:.0f}% of orders are single-SKU.\n"
        f"Weak Jaccard (median {median:.3f}) → turnover-based (ABC) storage "
        f"preferred over correlation-based."
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8.5, color="#333333")

    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    out_png = os.path.join(OUT_DIR, "08_sku_correlation_jaccard.png")
    fig.savefig(out_png, dpi=140)
    print(f"\nSaved chart -> {out_png}")
    print(f"Saved pairs -> {os.path.join(OUT_DIR, '08_sku_correlation_jaccard.csv')}")


if __name__ == "__main__":
    main()
