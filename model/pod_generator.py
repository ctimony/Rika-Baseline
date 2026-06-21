import csv
import random
import os
from pathlib import Path
import numpy as np
import pandas as pd

from model.pod_manager import PodManager


class PodGenerator:
    def __init__(self, pod_types, pod_num, total_sku, items_class_conf, items_pods_inventory_levels,
                 items_warehouse_inventory_levels, items_pods_class_conf, pod_manager: PodManager,
                 dev_mode=False):
        self.pod_types = pod_types
        self.total_sku = total_sku
        self.pod_num = pod_num
        self.items_class_conf = items_class_conf
        self.items_pods_inventory_level = items_pods_inventory_levels
        self.items_warehouse_inventory_levels = items_warehouse_inventory_levels
        self.items_pods_class_conf = items_pods_class_conf
        self.dev_mode = dev_mode

        self.pod_manager = pod_manager
    
    def config_items_slots(self, dev_mode=False):
        working_path = self.get_working_path(dev_mode)

        print("Setting up item slot configuration based on Item and Pod dictionary (see data/v2/ folder)...")

        # check if file not exists
        items_slots_path = working_path + "/items_slots_configuration.csv"
        if not os.path.exists(items_slots_path):

            pods_dictionary_path = working_path + "/pods_dictionary.csv"
            pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

            item_path = working_path + "/items_dictionary.csv"
            item_df = pd.read_csv(item_path, index_col=False)
            item_df = item_df[["item_code", "box_volume",
                            "item_volume", "number_of_item_in_a_box"]].copy()

            # getl slot type from pod_dictionary
            items_slots_configuration = pd.DataFrame()
            slot_types = np.sort(pods_dictionary["slot_type"].unique())
            for slot_type in slot_types:
                # get slot volume
                slot_volume = pods_dictionary.loc[pods_dictionary["slot_type"]
                                        == slot_type, "slot_volume"].values[0]
                item_df["slot_type"] = slot_type
                item_df["slot_volume"] = slot_volume
                item_df["max_box_in_slot"] = (
                    slot_volume / item_df["box_volume"]).astype(int)
                item_df["max_item_in_slot"] = item_df["max_box_in_slot"] * \
                    item_df["number_of_item_in_a_box"]

                items_slots_configuration = pd.concat(
                    [items_slots_configuration, item_df], axis=0)

            items_slots_configuration.to_csv(items_slots_path, index=False)

        else:
            print("    Item slot configuration already exists. Delete the file to reconfigure.")
            print()
    
    def check_items_pods_feasibility(self, total_sku, pod_types, pods_dictionary):
        feasible = True
        for pod_type in pod_types:
            total_slot = pods_dictionary.loc[pods_dictionary["pod_type"] == pod_type].shape[0]
            if  total_slot > total_sku:
                print("Pod type", pod_type, "has more slot ("+str(total_slot)+") than total SKU ("+str(total_sku)+").")
                print("In this system we asume that each pod cannot store same item in multiple slot.")
                print("Please adjust the number of items before we can continue to generate the items and pods.")
                feasible = False
                break
        return feasible
    
    def gen_items(self, pod_types=[3],
              total_sku=3000,
            #   items_class_conf={"A": 0.07, "B": 0.28, "C": 0.65},
              items_class_conf={"A": 0.171, "B": 0.389, "C": 0.440},
              items_pods_inventory_levels={"A": 0.3, "B": 0.4, "C": 0.5}, 
              items_warehouse_inventory_levels={"A": 0.3, "B": 0.4, "C": 0.5},
              select_option=1, 
              dev_mode=False):
        

        # Generate or select items based on class of items.
        # The items must align with the designated slot type within the pod.

        # get the working path
        working_path = self.get_working_path(dev_mode)

        # get item slot configuration
        items_slots_configuration_path = working_path + "/items_slots_configuration.csv"
        items_slots_configuration = pd.read_csv(items_slots_configuration_path, index_col=False)

        # get pod configuration
        pods_dictionary_path = working_path + "/pods_dictionary.csv"
        pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

        if self.check_items_pods_feasibility(total_sku, pod_types, pods_dictionary):
            
            # get list of item id that can be stored in the pod type (and slot type)
            item_code_arr = list()
            for pod_type in pod_types:
                slot_types = pods_dictionary.loc[pods_dictionary["pod_type"]
                                        == pod_type, "slot_type"].unique()
                slot_types = np.sort(slot_types)

                # print(slot_types)
                for slot_type in slot_types:
                    # print(slot_type)
                    item_code = items_slots_configuration.loc[(items_slots_configuration["slot_type"] == slot_type) & (
                        items_slots_configuration["max_box_in_slot"] > 0), "item_code"]
                    item_code_arr = item_code_arr + item_code.to_list()

            # remove duplicates
            item_code_arr = np.unique(item_code_arr)

            # retrieve item details corresponding to the list of item IDs compatible with the pod's slot specifications
            item_path = working_path + "/items_dictionary.csv"
            item_df = pd.read_csv(item_path, index_col=False)

            item_df = item_df.loc[item_df["item_code"].isin(item_code_arr)].copy()
            item_df = item_df.loc[item_df["slots_needed"] > 0].copy()
            item_df = item_df[["item_code", "item_class", "item_order_frequency", "item_initial_quantity_inventory",
                               "mean_daily_demand", "cv",
                               "box_length", "box_width", "box_height", "box_volume", "box_weight", "number_of_item_in_a_box",
                               "item_volume", "item_unit", "item_quantity_order_unique"]].copy()

            # ── SKU selection: read the final list from 06_sampling.py ───────────
            # Single source of truth. 06_sampling.py applies impact-quartile
            # stratified sampling (Opsi A+): impact = (composite × stability ×
            # variability)^(1/3) per SKU, split into quartiles within each ABC
            # class, A-Q4 taken in full (most important items never dropped),
            # other cells at one uniform fraction sized to the 489-pod budget.
            # This selection is DETERMINISTIC (sort + head, no RNG) so items.csv
            # is identical across replications — only orders/robots vary.
            # pod_generator does NOT re-sample; it only places the chosen SKUs,
            # intersecting with item_df (SKUs compatible with the pod slot type).
            sampled_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "data_mining", "output", "06_sampled_skus.csv")
            sampled = pd.read_csv(sampled_path)
            sampled["item_code"] = sampled["item_code"].astype(item_df["item_code"].dtype)
            sampled_codes = set(sampled["item_code"])

            items = item_df.loc[item_df["item_code"].isin(sampled_codes)].copy()
            # Per-class inventory-level tags (used downstream by placement)
            items["item_pod_inventory_level"] = items["item_class"].map(items_pods_inventory_levels)
            items["item_warehouse_inventory_level"] = items["item_class"].map(items_warehouse_inventory_levels)
            # Keep ABC ordering then by frequency for stable, reproducible output
            items = items.sort_values(["item_class", "item_order_frequency"],
                                      ascending=[True, False])

            dropped = sampled_codes - set(items["item_code"])
            if dropped:
                print(f"    WARNING: {len(dropped)} sampled SKUs are not slot-compatible "
                      f"and were skipped (of {len(sampled_codes)} selected).")
            print(f"    Loaded {len(items)} SKUs from 06_sampled_skus.csv "
                  f"(A={int((items['item_class']=='A').sum())}, "
                  f"B={int((items['item_class']=='B').sum())}, "
                  f"C={int((items['item_class']=='C').sum())})")

            # save items selected to csv
            items.reset_index(drop=True, inplace=True)
            items.index.name = "item_id"

            # item_weight UNIFORM = 0.2 kg/unit for all SKUs: controls the experiment so pod
            # mass (Σ item_weight × current_qty) reflects ONLY the number of units carried,
            # not per-SKU weight differences. This isolates the replenishment-policy effect on
            # energy (pod mass) — variation in mass comes purely from how many units a trip
            # moves (policy/fill-rate), not from which SKU is heavy. 0.2 is a realistic small-
            # moving-item weight (within the original distribution, p25≈0.18) → pod mass ~200–
            # 680 kg, matching real RMFS pod loads, instead of the inflated values from w=1.
            items.insert(11, "item_weight", 0.2)
            items[["item_order_frequency", "number_of_item_in_a_box"]] = items[[
                "item_order_frequency", "number_of_item_in_a_box"]].astype(int)

            items.to_csv(working_path + "/items.csv", index=True)
            
        else:
            items = None

        return items

    def gen_pods(self, pod_num, pod_types, dev_mode=True):

        # get the working path
        working_path = self.get_working_path(dev_mode)

        # get pod configuration
        pods_dictionary_path = working_path + "/pods_dictionary.csv"
        pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

        # generate pods based on the pod specification (types and the number of pods)
        counter_id = 0
        pods = pd.DataFrame()
        for i, p in enumerate(pod_types):

            pod_slot_num = pods_dictionary.loc[pods_dictionary["pod_type"] == p].shape[0]
            pod_slot_id = pods_dictionary.loc[pods_dictionary["pod_type"]
                                    == p, "slot_sequence"].to_list()
            slot_type = pods_dictionary.loc[pods_dictionary["pod_type"]
                                    == p, "slot_type"].to_list()       

            pod_type = [p] * pod_slot_num
            pod_face = pods_dictionary.loc[pods_dictionary["pod_type"] == p, "pod_face"].to_list()

            for _ in range(pod_num[i]):
                pod_id = [counter_id] * pod_slot_num
                counter_id += 1

                pod = pd.DataFrame({"pod_id": pod_id,
                                    "pod_type": pod_type,
                                    "slot_id": pod_slot_id,
                                    "slot_type": slot_type,
                                    "item": np.nan,
                                    "unusedColumn1": 0,
                                    "unusedColumn2": 0,
                                    "unusedColumn3": 0,
                                    "qty": np.nan,
                                    "max_qty": np.nan,
                                    'due_date': 99999,
                                    'facing': pod_face,
                                    'pick_ind': 0,
                                    'item_weight': 0.0,
                                    'total_item_weight': 0.0,
                                    })
                pods = pd.concat([pods, pod], axis=0)

        pods.reset_index(drop=True, inplace=True)
        return pods
    
    def assign_items_to_pods(self, pods, items, items_pods_class_conf,
                             class_slot_counts={"A": 10, "B": 5, "C": 5},
                             dev_mode=False):

        working_path = self.get_working_path(dev_mode)

        # ── Phase 1: Prepare ──────────────────────────────────────────────────
        pods_dictionary_path = working_path + "/pods_dictionary.csv"
        pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)
        slot_types = pods_dictionary.loc[
            pods_dictionary["pod_type"].isin(pods["pod_type"].unique()),
            "slot_type"
        ].unique()

        item_slot_cfg_path = working_path + "/items_slots_configuration.csv"
        items_slots_cfg = pd.read_csv(item_slot_cfg_path, index_col=False)
        items_slots_cfg_sel = items_slots_cfg.loc[
            (items_slots_cfg["item_code"].isin(items["item_code"].unique())) &
            (items_slots_cfg["slot_type"].isin(slot_types)) &
            (items_slots_cfg["max_item_in_slot"] > 0)
        ].copy()

        items["item_id"] = items.index
        items_slots_cfg_sel = items_slots_cfg_sel.merge(
            items[["item_id", "item_code", "item_class", "item_weight",
                   "item_initial_quantity_inventory",
                   "item_pod_inventory_level", "item_warehouse_inventory_level"]].copy(),
            how="inner", on="item_code"
        )

        # One row per item: use the slot_type present in pods (take first match)
        items_to_place = (
            items_slots_cfg_sel
            .sort_values(["item_class", "item_id"])
            .drop_duplicates(subset="item_id", keep="first")
            [["item_id", "item_code", "item_class", "max_item_in_slot",
              "item_weight", "item_initial_quantity_inventory",
              "item_pod_inventory_level", "item_warehouse_inventory_level"]]
            .copy()
        )
        items_to_place = items_to_place[
            items_to_place["item_initial_quantity_inventory"] > 0
        ].copy()
        # Attach order_frequency (from items.csv) so the assignment can place hot SKUs
        # first → hot-grouping. Default to 0 if a SKU is missing it.
        _items_csv = pd.read_csv(working_path + "/items.csv")
        if "item_order_frequency" in _items_csv.columns:
            _freq = dict(zip(_items_csv["item_code"], _items_csv["item_order_frequency"]))
            items_to_place["item_order_frequency"] = (
                items_to_place["item_code"].map(_freq).fillna(0)
            )
        else:
            items_to_place["item_order_frequency"] = 0

        pods["slot_sequence"] = np.arange(pods.shape[0])

        # ── Phase 2: Build per-pod slot pools per class (INTERLEAVED) ──────────
        # The per-pod class quota is NOT a single fixed integer (the ideal quota is
        # fractional, e.g. A=9.57, B=5.48, C=4.95 per pod, and any single integer
        # over/under-allocates a class → some pods miss a class). Instead the quota
        # is sized to the ACTUAL slot supply per class and distributed with the
        # LARGEST-REMAINDER (Hamilton) method: every pod gets floor(ideal) slots of
        # each class, then the leftover slots (so the per-class total equals supply
        # exactly) are handed one each to the pods with the largest fractional
        # remainder. This guarantees total(pool[cls]) == slots_needed[cls], no slot
        # is over/short, and — because each pod's quota for every class is ≥ floor —
        # every pod still receives all three classes (pods stay mixed A/B/C).
        # The pool is kept PER POD (not flattened) so the interleaved assignment
        # can take at most one slot per pod for any single SKU.
        classes = list(class_slot_counts.keys())  # class order: A, B, C
        n_pods = pods["pod_id"].nunique()
        # Supply = number of slots each class needs (Σ slots_needed over its SKUs).
        supply = {}
        for cls in classes:
            cls_rows = items_to_place[items_to_place["item_class"] == cls]
            need = np.ceil(cls_rows["item_initial_quantity_inventory"]
                           / cls_rows["max_item_in_slot"].clip(lower=1)).astype(int)
            supply[cls] = int(need.sum())

        # 2D largest-remainder: each pod's per-class quota must sum to that pod's
        # PHYSICAL slot count (slots_per_pod), AND each class's column total must
        # equal supply[cls]. (Allocating each class independently breaks the row
        # constraint — a pod could be assigned 21 quota for 20 real slots → the
        # interleaved walk pops from an empty slot list.) So: give every pod the
        # floor of each class's ideal share, then hand out the per-pod leftover
        # (slots_per_pod − Σ floors) one slot at a time to the classes with the
        # largest fractional remainder, decrementing a global per-class budget so
        # the column totals stay exactly = supply. Result: rows sum to 20, columns
        # sum to supply, all SKUs of every class get a home in distinct pods.
        slots_per_pod = pods.groupby("pod_id").size().iloc[0]
        ideal = {cls: supply[cls] / n_pods for cls in classes}
        base = {cls: int(np.floor(ideal[cls])) for cls in classes}
        frac = {cls: ideal[cls] - base[cls] for cls in classes}
        # remaining +1 slots still owed to each class so its column total = supply
        col_budget = {cls: supply[cls] - base[cls] * n_pods for cls in classes}
        per_pod_quota = {cls: [base[cls]] * n_pods for cls in classes}
        per_pod_leftover = slots_per_pod - sum(base.values())  # +slots each pod needs
        # Hand the per-pod leftovers to the highest-remainder classes that still
        # have column budget; tie-break by class order. Even spreading isn't needed
        # because every pod gets the same `per_pod_leftover` and frac is identical
        # across pods, so the budget drains uniformly class-by-class.
        order = sorted(classes, key=lambda c: (-frac[c], classes.index(c)))
        for pod_pos in range(n_pods):
            need = per_pod_leftover
            for cls in order:
                if need <= 0:
                    break
                if col_budget[cls] > 0:
                    per_pod_quota[cls][pod_pos] += 1
                    col_budget[cls] -= 1
                    need -= 1
            # if the highest-remainder classes are exhausted, fall back to any
            # class with remaining budget (keeps the row sum at slots_per_pod)
            if need > 0:
                for cls in classes:
                    while need > 0 and col_budget[cls] > 0:
                        per_pod_quota[cls][pod_pos] += 1
                        col_budget[cls] -= 1
                        need -= 1
        assert all(col_budget[c] == 0 for c in classes), \
            f"column budget not drained: {col_budget}"
        assert all(
            sum(per_pod_quota[c][p] for c in classes) == slots_per_pod
            for p in range(n_pods)
        ), "a pod's quota does not sum to its physical slot count"

        print("    Building proportional slot pools — interleaved, largest-remainder "
              "(supply A=%d B=%d C=%d over %d pods)..." % (
                  supply.get("A", 0), supply.get("B", 0), supply.get("C", 0), n_pods))

        # Partition each pod's empty slots into per-class blocks using that pod's
        # own quota, keyed PER POD (not flattened) so the interleaved assignment
        # below can draw at most one slot per pod for any single SKU.
        pod_class_slots = {cls: {} for cls in classes}  # cls → {pod_id: [slot idx]}
        pod_class_cap   = {cls: {} for cls in classes}  # cls → {pod_id: remaining cap}
        for pod_pos, (pid, grp) in enumerate(pods[pods["item"].isnull()].groupby("pod_id")):
            indices = grp.index.tolist()
            cursor = 0
            for cls in classes:
                count = per_pod_quota[cls][pod_pos]
                pod_class_slots[cls][pid] = indices[cursor:cursor + count]
                pod_class_cap[cls][pid]   = count
                cursor += count

        # ── Phase 3: Main assignment — class by class, STACKED ────────────────
        # STACKED storage: a SKU's `slots_needed` slots may be taken from the SAME
        # pod (one SKU can occupy multiple slots in one pod), in contrast to the
        # interleaved version which spread a SKU across distinct pods. Stacked
        # makes a pod arrive at replenishment substantially emptier (many slots of
        # the triggering SKU are depleted together), so the fill-level decision has
        # a meaningful effect on pod mass and energy.
        #
        # Within each class, SKUs are placed largest-first; each SKU greedily fills
        # whole pods from the per-pod class pool (consuming as many of that pod's
        # class slots as fit) before moving to the next pod. The per-pod class
        # quota from Phase 2 is preserved, so each pod still receives a mixed A/B/C
        # composition. Per-class supply == Σ slots_needed, so allocation is exact.
        #
        # ── INTERLEAVED (previous version — kept for reference) ────────────────
        # unplaced_ids = set()
        # for cls in classes:
        #     cls_items = items_to_place[items_to_place["item_class"] == cls].copy()
        #     cls_items["__slots_needed"] = np.ceil(
        #         cls_items["item_initial_quantity_inventory"]
        #         / cls_items["max_item_in_slot"].clip(lower=1)
        #     ).astype(int)
        #     cls_items = cls_items.sort_values(
        #         ["__slots_needed", "item_id"], ascending=[False, True]
        #     )
        #     remaining_cap = dict(pod_class_cap[cls])  # pod_id → remaining capacity
        #     print(f"    Class {cls}: {len(cls_items)} SKUs → "
        #           f"{sum(remaining_cap.values())} pool slots (interleaved)")
        #
        #     for _, row in cls_items.iterrows():
        #         max_fit = int(row["max_item_in_slot"])
        #         if max_fit <= 0:
        #             max_fit = 1
        #         slots_needed = int(row["__slots_needed"])
        #
        #         # Most-free-first: pick the slots_needed pods with the largest
        #         # remaining capacity (tie-break by pod_id for determinism).
        #         live = sorted((p for p, c in remaining_cap.items() if c > 0),
        #                       key=lambda p: (-remaining_cap[p], p))[:slots_needed]
        #         if len(live) < slots_needed:
        #             # Not enough distinct pods with free capacity → defer to repair.
        #             unplaced_ids.add(int(row["item_id"]))
        #             continue
        #
        #         chosen = []
        #         for p in live:
        #             chosen.append(pod_class_slots[cls][p].pop())  # one slot, distinct pod
        #             remaining_cap[p] -= 1
        #
        #         pods.loc[chosen, "item"]                          = int(row["item_id"])
        #         pods.loc[chosen, "qty"]                           = max_fit
        #         pods.loc[chosen, "max_qty"]                       = max_fit
        #         pods.loc[chosen, "item_weight"]                   = row["item_weight"]
        #         pods.loc[chosen, "total_item_weight"]             = round(row["item_weight"] * max_fit, 3)
        #         pods.loc[chosen, "item_pod_inventory_level"]      = row["item_pod_inventory_level"]
        #         pods.loc[chosen, "item_warehouse_inventory_level"]= row["item_warehouse_inventory_level"]
        #
        # # ── Phase 4 (interleaved): Feasibility guard ──────────────────────────
        # if unplaced_ids:
        #     raise RuntimeError(
        #         f"Interleaved allocation failed: {len(unplaced_ids)} SKUs could "
        #         f"not be placed in distinct pods (item_ids: {sorted(unplaced_ids)}). "
        #         f"Check per-class supply vs. max slots_needed."
        #     )
        # ───────────────────────────────────────────────────────────────────────

        # ── Phase 3: Main assignment — class by class, CAPPED-STACKED ─────────
        # CAPPED-STACKED storage: a SKU may take up to STACK_CAP (=3) slots of the
        # SAME pod, then must spill to the next pod. This balances two competing aims:
        #   • buffer (trips): up to 3 slots of one SKU per pod → one trip refills a
        #     meaningful chunk → the SKU is re-triggered less often → fewer trips.
        #   • opportunity-score choice: a high-demand class-A SKU (slots_needed > 3)
        #     spans ⌈slots_needed/3⌉ DISTINCT pods, so when it goes critical the
        #     replenishment pod-selection actually has several candidate pods to rank
        #     — preserving the score's role (which full stacking, one-pod-per-SKU,
        #     destroys: 93% of class-A SKUs would live in a single pod).
        # Class B/C SKUs (slots_needed ≈ 1) still land in one slot each. The per-pod
        # class quota from Phase 2 is preserved → pods stay mixed A/B/C. Per-class
        # supply == Σ slots_needed, so allocation is exact.
        #
        # Feasibility: with cap, a SKU needs ⌈slots_needed/cap⌉ DISTINCT pods that
        # still have free class capacity. The largest class-A SKU needs
        # ⌈max_slots_needed / 3⌉ pods ≪ 489, and every pod's class quota ≥ 1, so the
        # most-free-first greedy always finds enough distinct pods (no dead-end). The
        # guard below raises if any SKU is left short rather than silently dropping it.
        # ── INTERLEAVED + HOT-GROUPING ─────────────────────────────────────────
        # INTERLEAVED: each SKU takes at most ONE slot per pod (a SKU's slots_needed
        # slots go to distinct pods), so every SKU spreads across many pods → the
        # opportunity-score pod-selection has real candidate pods to rank.
        # HOT-GROUPING: within each class, SKUs are placed in DESCENDING order_frequency
        # (hottest first), and pods are filled in a FIXED order, each packed to its class
        # quota before moving on. So the highest-frequency ('hot') SKUs of a class land in
        # the SAME early pods → those pods become 'hot pods' carrying many fast-moving
        # SKUs across all classes. Hot pods go critical more often, and when fast-movers
        # that co-occur in an order are picked together they can present several critical
        # SKUs at once → the opportunity score can cover multiple critical SKUs per trip.
        # The per-pod class quota from Phase 2 is UNCHANGED, so every pod stays mixed
        # A/B/C (hot-grouping only reorders SKUs WITHIN each class, never the class mix).
        unplaced_ids = set()
        for cls in classes:
            cls_items = items_to_place[items_to_place["item_class"] == cls].copy()
            cls_items["__slots_needed"] = np.ceil(
                cls_items["item_initial_quantity_inventory"]
                / cls_items["max_item_in_slot"].clip(lower=1)
            ).astype(int)
            # Hottest SKUs first so they cluster into the same (early) pods.
            cls_items = cls_items.sort_values(
                ["item_order_frequency", "item_id"], ascending=[False, True]
            )
            # Fill pods in a FIXED order (by pod_id) so consecutive hot SKUs share pods.
            pod_order = sorted(pod_class_cap[cls].keys())
            remaining_cap = dict(pod_class_cap[cls])  # pod_id → remaining capacity
            print(f"    Class {cls}: {len(cls_items)} SKUs → "
                  f"{sum(remaining_cap.values())} pool slots (interleaved + hot-grouping)")

            for _, row in cls_items.iterrows():
                max_fit = int(row["max_item_in_slot"])
                if max_fit <= 0:
                    max_fit = 1
                slots_needed = int(row["__slots_needed"])

                # Interleaved: one slot per distinct pod. Walk pods in fixed order, take
                # exactly ONE slot from each pod that still has class capacity, until
                # slots_needed distinct pods are filled. Fixed pod order (not most-free)
                # makes hot SKUs accumulate in the same early pods → hot pods form.
                chosen = []
                need = slots_needed
                for p in pod_order:
                    if need <= 0:
                        break
                    if remaining_cap[p] <= 0:
                        continue
                    slot_list = pod_class_slots[cls][p]
                    chosen.append(slot_list.pop())       # one slot, this pod
                    remaining_cap[p] -= 1
                    need -= 1

                if need > 0:
                    # Fixed-order pass left this SKU short (its slots_needed exceeded the
                    # pods still holding class capacity in id-order). Fall back to ANY
                    # remaining pods with capacity (most-free-first, distinct) so it is
                    # placed in full. This only affects the few large-slots_needed SKUs;
                    # the hot-grouping ordering still holds for the common 1-slot SKUs.
                    extra = sorted((p for p, c in remaining_cap.items() if c > 0),
                                   key=lambda p: (-remaining_cap[p], p))
                    for p in extra:
                        if need <= 0:
                            break
                        slot_list = pod_class_slots[cls][p]
                        if not slot_list:
                            continue
                        chosen.append(slot_list.pop())
                        remaining_cap[p] -= 1
                        need -= 1

                if need > 0:
                    # Truly not enough distinct pods with free class capacity → guard.
                    unplaced_ids.add(int(row["item_id"]))
                    continue

                pods.loc[chosen, "item"]                          = int(row["item_id"])
                pods.loc[chosen, "qty"]                           = max_fit
                pods.loc[chosen, "max_qty"]                       = max_fit
                pods.loc[chosen, "item_weight"]                   = row["item_weight"]
                pods.loc[chosen, "total_item_weight"]             = round(row["item_weight"] * max_fit, 3)
                pods.loc[chosen, "item_pod_inventory_level"]      = row["item_pod_inventory_level"]
                pods.loc[chosen, "item_warehouse_inventory_level"]= row["item_warehouse_inventory_level"]

        # ── Phase 4: Feasibility guard ─────────────────────────────────────────
        # Per-class supply == Σ slots_needed, so the pool holds exactly enough class
        # slots for every SKU of that class; unplaced_ids must be empty. If non-empty,
        # a class ran out of distinct pods before all its SKUs were placed — raise
        # rather than silently leaving SKUs (incl. item_id 0) unallocated.
        if unplaced_ids:
            raise RuntimeError(
                f"Interleaved + hot-grouping allocation failed: {len(unplaced_ids)} SKUs "
                f"could not be placed in distinct pods (item_ids: {sorted(unplaced_ids)}). "
                f"Check per-class supply vs. max slots_needed."
            )

        # ── Phase 5: Finalize and save ────────────────────────────────────────
        # Empty slots: item = -1 (NOT 0, which would collide with the real item_id 0);
        # qty/max_qty = 0. This makes empty slots unambiguously identifiable.
        pods["item"] = pods["item"].fillna(-1).astype(int)
        pods[["qty", "max_qty"]] = pods[["qty", "max_qty"]].fillna(0).astype(int)
        pods = pods.sort_values(["pod_id", "slot_sequence"]).reset_index(drop=True)
        pods["cumulative_pod_weight"] = pods.groupby("pod_id")["total_item_weight"].cumsum().round(3)

        assigned = (pods["item"] > 0).sum()
        print(f"    Assignment complete: {assigned:,} slots filled / {len(pods):,} total")

        pods.to_csv(working_path + "/pods.csv", index=False)
        return pods
    
    def config_items_pods(self, pod_types=[3], pod_num=[377], total_sku=3000,
                    #   items_class_conf={"A": 0.07, "B": 0.28, "C": 0.65},
                      items_class_conf={"A": 0.171, "B": 0.389, "C": 0.440},
                      items_pods_inventory_levels={"A": 0.4, "B": 0.5, "C": 0.6},
                      items_warehouse_inventory_levels={"A": 0.3, "B": 0.4, "C": 0.5},
                    #   items_pods_class_conf={"A": 0.7, "B": 0.2, "C": 0.1}, # original
                      items_pods_class_conf={"A": 0.7, "B": 0.1, "C": 0.2}, # chat gpt data 13
                      dev_mode=False):
        
        working_path = self.get_working_path(dev_mode)
        
        self.config_items_slots(dev_mode=dev_mode)
        items_path = working_path + "/items.csv"
        print("Configuring items...")
        
        if not os.path.exists(items_path):
            print("    Items configuration is not found. We will generate the items based on the class configuration.")
            items = self.gen_items(pod_types=pod_types, 
                            total_sku=total_sku, 
                            items_class_conf=items_class_conf,
                            items_pods_inventory_levels=items_pods_inventory_levels,  
                            items_warehouse_inventory_levels=items_warehouse_inventory_levels,
                            select_option=0,
                            dev_mode=dev_mode)       
            items_flag = True
        else:
            items = pd.read_csv(items_path, index_col=0)

            if items.shape[0] == total_sku:
                print("    Items already exist and the number of items is the same as the total SKU.")
                print("    We will use the existing items file.")
                items_flag = False

            elif items.shape[0] > total_sku:
                print("    Items already exist but the number of items is more than the total SKU.")
                print("    We will use the existing items file and select the items based on the total SKU and the class configuration.")
                for k, v in items_class_conf.items():
                    items_class = items.loc[items["item_class"]==k]
                    items_class = items_class.sort_values(by="item_order_frequency", ascending=False)
                    items_class = items_class.head(int(total_sku*v))
                    items = pd.concat([items, items_class])
                    items = items.drop_duplicates(subset="item_code")
                    if items.shape[0] == total_sku:
                        items.reset_index(drop=True, inplace=True)
                        break
                items.to_csv(items_path, index=True)
                items_flag = True
            else:
                print("    Items already exist but the number of items is less than the total SKU.")
                print("    We will generate the items based on number of items needed and the class configuration.")
                items = self.gen_items(pod_types=pod_types, 
                                total_sku=total_sku, 
                                items_class_conf=items_class_conf,
                                items_pods_inventory_levels=items_pods_inventory_levels,  
                                items_warehouse_inventory_levels=items_warehouse_inventory_levels,
                                select_option=0,
                                dev_mode=dev_mode)   
                items_flag = True
            print("    Items configuration is done. If you want to reconfigure the items, please delete the items.csv file.")
            print()

        if items is not None:

            pods_path = working_path + "/pods.csv"
            
            print("Configuring pods and assigning items to the pods...")
            if not os.path.exists(pods_path):

                print("    Pods configuration is not found. We will generate the pods based on the configuration.")
                pods = self.gen_pods(pod_types=pod_types, pod_num=pod_num, dev_mode=dev_mode)
                pods = self.assign_items_to_pods(pods, items, items_pods_class_conf=items_pods_class_conf, dev_mode=dev_mode)
                pods_flag = True

            else:
                pods = pd.read_csv(pods_path, index_col=False)
                pod_list = pods["pod_id"].unique().tolist()
                pod_type_list = pods["pod_type"].unique().tolist()

                flag_pod_num_per_type = True
                for i, pod_type in enumerate(pod_types):
                    if pod_num[i] != len(pods.loc[pods["pod_type"]==pod_type, "pod_id"].unique().tolist()):
                        flag_pod_num_per_type = False
                        break
                
                flag_pod_type = True
                for pod_type in pod_type_list:
                    if pod_type not in pod_types:
                        flag_pod_type = False
                        break
                        
                flag_pod_num = True
                if len(pod_list) != sum(pod_num):
                    flag_pod_num = False

                if flag_pod_num_per_type and flag_pod_type and flag_pod_num:
                    print("    Pods already exist and the number of pods, number of pod per pods type, and the pod type are the same as the configuration.")
                    print("    We will use the existing pods file.")
                    pods_flag = False
                else:
                    print("    Pods already exist but the number of pods, number of pod per pods type, or the pod type is different from the configuration.")
                    print("    We will generate the pods based on the configuration.")
                    pods = self.gen_pods(pod_types=pod_types, pod_num=pod_num, dev_mode=dev_mode)
                    pods = self.assign_items_to_pods(pods, items, items_pods_class_conf=items_pods_class_conf, dev_mode=dev_mode)
                    pods_flag = True

            print("    Pods configuration is done. If you want to reconfigure the pods, please delete the pods.csv file.")
            print()
        else:     
            pods = None

        return     
    

    def generate(self):
        self.config_items_pods(self.pod_types, self.pod_num, self.total_sku, self.items_class_conf, self.items_pods_inventory_level
                               ,self.items_warehouse_inventory_levels, self.items_pods_class_conf, self.dev_mode)
    
    def get_working_path(self,dev_mode=False):

        if dev_mode:
            # development mode

            # get the parent directory
            p = Path(__file__).parents[2]

            # p to string
            p = str(p)

            result = p
        else:
            # production/NetLogo mode

            # get the parent directory
            result = os.getcwd()

        return result

