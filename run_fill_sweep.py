"""
run_fill_sweep.py
=================
Sweep V12_FILL_RATE for a SINGLE lead time (whatever rop_summary.csv currently
encodes). Runs the full simulation once per fill rate, back to back, and labels
each result folder + summary.csv with the fill rate — so you never have to guess
which folder is which.

Lead time is NOT changed here (changing L needs rop_summary regeneration). Set L
and regenerate rop_summary FIRST, then sweep fill rate with this script.

Usage
-----
  python3 run_fill_sweep.py              # ALL: fill rates 0.8, 0.9, 1.0  (default)
  python3 run_fill_sweep.py 0.8 0.9      # only 0.8 and 0.9
  python3 run_fill_sweep.py 1.0          # only 1.0
  python3 run_fill_sweep.py 0.7 0.85 1.0 # any custom set

Output
------
  result/<timestamp>_fill<rate>/        # one folder per fill rate (rate in name)
      summary.csv                       # with an added 'fill_rate' first column
      pod_info.csv, tick_metrics.csv, ...
  fill_sweep_summary.csv                # combined table across the sweep (root)

The SAME generated_order.csv on disk is reused for every fill rate, so the runs
are PAIRED (only fill rate differs) — clean for a paired comparison. Delete
generated_order.csv first if you want a fresh order set.
"""

import os
import sys
import glob
import time

import pandas as pd

import model.inventory as inv   # import the MODULE so we can override the global
import netlogo

# Default = ALL fill rates swept when no argument is given.
DEFAULT_FILLS = [0.8, 0.9, 1.0]


def _newest_result_dir(before):
    """Return the result/ folder created since the `before` snapshot (a set)."""
    after = set(glob.glob(os.path.join("result", "*")))
    new = after - before
    if not new:
        return None
    return max(new, key=os.path.getmtime)   # most recently modified new folder


def _clean_accumulating_state():
    """Delete output/order-finished.csv before a run. It is written in APPEND mode
    (model/inventory.py:write_to_csv -> output/) and is NEVER reset by setup(), so
    without this it carries order rows over from the previous fill rate and corrupts
    the throughput metric (netlogo.py counts nunique order_id from this file). All
    other state files ARE reset by setup(), so this is the only one we must clear.
    Inputs (generated_order.csv, rop_summary.csv) and result/ are left untouched."""
    stale = os.path.join("output", "order-finished.csv")
    if os.path.exists(stale):
        os.remove(stale)
        print(f"  cleaned stale {stale} (would otherwise accumulate)")


def run_one(fill_rate):
    """Run one full 8-h simulation at `fill_rate`. Returns the tagged result dir."""
    inv.V12_FILL_RATE = fill_rate           # read live at replenishment time
    print("=" * 64)
    print(f"  RUN  fill_rate = {fill_rate}   ORS_VERSION = {inv.ORS_VERSION}")
    print("=" * 64)

    _clean_accumulating_state()             # fresh order-finished.csv per fill rate

    before = set(glob.glob(os.path.join("result", "*")))

    # same two calls you use by hand in the terminal:
    netlogo.setup()
    netlogo.console_tick()

    rd = _newest_result_dir(before)
    if rd is None:
        print(f"  WARNING: no new result folder found for fill={fill_rate}")
        return None

    # tag the folder with the fill rate so it is self-labelling
    tagged = rd + f"_fill{fill_rate}"
    if os.path.exists(tagged):
        tagged = f"{tagged}_{int(time.time())}"
    os.rename(rd, tagged)

    # add fill_rate as a column in summary.csv (in the data, not just the name)
    summ = os.path.join(tagged, "summary.csv")
    if os.path.exists(summ):
        s = pd.read_csv(summ)
        s.insert(0, "fill_rate", fill_rate)
        s.to_csv(summ, index=False)
        r = s.iloc[0]
        en = r.get("total_energy")
        oc = r.get("orders_finished")
        eo = round(en / oc, 2) if (isinstance(en, (int, float)) and oc) else "?"
        print(f"  -> {tagged}")
        print(f"     throughput={r.get('order_throughput')}  energy={en}  "
              f"energy/order={eo}  trips={r.get('total_replenishments')}  "
              f"skus_repl={r.get('skus_replenished')}  "
              f"stockout={r.get('stockout_count')}")
    else:
        print(f"  -> {tagged}  (summary.csv not found)")
    return tagged


def _worker(fill_rate):
    """Run ONE fill rate in THIS (fresh) process and print the tagged dir on the
    last line so the parent can collect it. Must run in its own process: the sim
    keeps module/class-level global state (e.g. RobotJob.counter, written CSVs)
    that is NOT reset between runs, so doing >1 run in one process corrupts the
    2nd+ run (order ids drift → get_order_by_id returns None → crash). One process
    per fill = identical to running it by hand in a fresh terminal."""
    rd = run_one(fill_rate)
    # sentinel line the parent greps for:
    print(f"__RESULT_DIR__={rd if rd else ''}")


def main():
    # Worker mode: `python3 run_fill_sweep.py --worker 0.8` runs a single fill in
    # its own process. The parent (normal invocation) spawns one worker per fill.
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(float(sys.argv[2]))
        return

    import subprocess

    fills = [float(x) for x in sys.argv[1:]] or list(DEFAULT_FILLS)
    print(f"\nFill-rate sweep (single lead time): {fills}")
    print("Lead time = whatever rop_summary.csv currently encodes "
          "(regenerate it first to change L).")
    print("Each fill runs in its OWN process (no global-state carry-over).\n")

    produced = []
    for fr in fills:
        print(f"\n>>> launching fresh process for fill_rate={fr} ...")
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", str(fr)],
            capture_output=True, text=True,
        )
        # stream the worker's output through so you still see the sim log
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            print(f"  !! fill_rate={fr} FAILED (exit {proc.returncode}) — skipping, "
                  f"sweep continues with remaining fills.")
            continue
        # parse the sentinel for the tagged result dir
        rd = None
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT_DIR__="):
                rd = line.split("=", 1)[1].strip() or None
        if rd and os.path.exists(os.path.join(rd, "summary.csv")):
            produced.append(rd)
        else:
            print(f"  !! fill_rate={fr} produced no summary.csv — skipping.")

    # combined table for easy comparison across the sweep
    rows = [pd.read_csv(os.path.join(rd, "summary.csv"))
            for rd in produced
            if os.path.exists(os.path.join(rd, "summary.csv"))]
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv("fill_sweep_summary.csv", index=False)
        cols = [c for c in ["fill_rate", "order_throughput", "total_energy",
                            "orders_finished", "total_replenishments",
                            "skus_replenished", "trip_efficiency",
                            "stockout_count"]
                if c in combined.columns]
        print("\n" + "=" * 64)
        print("  SWEEP DONE — combined results -> fill_sweep_summary.csv")
        print(combined[cols].to_string(index=False))
        print("=" * 64)


if __name__ == "__main__":
    main()
