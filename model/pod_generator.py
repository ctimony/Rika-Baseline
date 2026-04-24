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
                 pod_wmax: float = 600.0, dev_mode=False):
        self.pod_types = pod_types
        self.total_sku = total_sku
        self.pod_num = pod_num
        self.items_class_conf = items_class_conf
        self.items_pods_inventory_level = items_pods_inventory_levels
        self.items_warehouse_inventory_levels = items_warehouse_inventory_levels
        self.items_pods_class_conf = items_pods_class_conf
        self.pod_wmax = pod_wmax
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
    
    def gen_items(self, pod_types=[0], 
              total_sku=500, 
            #   items_class_conf={"A": 0.07, "B": 0.28, "C": 0.65}, 
              items_class_conf={"A": 0.1, "B": 0.3, "C": 0.6},
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
            item_df = item_df[["item_code", "item_class", "item_order_frequency", "item_initial_quantity_inventory", "box_length", "box_width", "box_height", "box_volume", "box_weight", "number_of_item_in_a_box",
                            "item_volume", "item_unit", "item_quantity_order_unique"]].copy()

            # select items according to the class and its proportion
            items = pd.DataFrame()
            for class_name, conf in items_class_conf.items():

                # get list of the item's code based on the class
                item = item_df.loc[item_df["item_class"] == class_name].copy()
                item = item.sort_values(by="item_order_frequency", ascending=False)

                # print(items_inventory_levels)
                item["item_pod_inventory_level"] = items_pods_inventory_levels[class_name]
                item["item_warehouse_inventory_level"] = items_warehouse_inventory_levels[class_name]

                # calculate the probability of order frequency for each item
                item_probability = item["item_order_frequency"] / \
                    item["item_order_frequency"].sum()

                # select option 1: select items based on the probability of order frequency for each class.
                # else: select items based on the top n-percent of the class ratio
                if select_option == 1:

                    # select items based on the probability of order frequency for each class.
                    item_code = np.random.choice(item["item_code"].to_list(), size=int(
                        total_sku*conf), p=item_probability, replace=False)
                    items = pd.concat(
                        [items, item.loc[item["item_code"].isin(item_code)]])
                else:

                    # select items based on the top n-percent of the class ratio
                    item_top = item.head(int(total_sku*conf))
                    items = pd.concat([items, item_top])

            # save items selected to csv
            items.reset_index(drop=True, inplace=True)
            items.index.name = "item_id"

            items_weight = items["box_weight"] / items["number_of_item_in_a_box"]
            items.insert(11, "item_weight", items_weight.round(3))
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
                             class_slot_counts={"A": 7, "B": 6, "C": 7},
                             dev_mode=False):
        working_path = self.get_working_path(dev_mode)

        # ----------------------------------------------------------------
        # 1. Load slot configuration and merge with item attributes
        # ----------------------------------------------------------------
        pods_dictionary = pd.read_csv(working_path + "/pods_dictionary.csv", index_col=False)
        slot_types = (
            pods_dictionary
            .loc[pods_dictionary["pod_type"].isin(pods["pod_type"])]
            .sort_values("slot_volume", ascending=False)["slot_type"]
            .unique()
        )

        pods["slot_sequence"] = np.arange(len(pods))

        items_slots_cfg = pd.read_csv(working_path + "/items_slots_configuration.csv", index_col=False)
        items_slots_cfg = items_slots_cfg.loc[
            items_slots_cfg["item_code"].isin(items["item_code"]) &
            items_slots_cfg["slot_type"].isin(slot_types) &
            (items_slots_cfg["max_box_in_slot"] > 0)
        ]

        items["item_id"] = items.index
        items_slots_cfg = items_slots_cfg.merge(
            items[["item_id", "item_code", "item_class", "box_weight", "item_weight",
                   "item_order_frequency", "item_initial_quantity_inventory",
                   "item_pod_inventory_level", "item_warehouse_inventory_level"]].copy(),
            how="inner", on="item_code"
        )
        items_slots_cfg["slots_needed"] = np.ceil(
            items_slots_cfg["item_initial_quantity_inventory"] /
            items_slots_cfg["number_of_item_in_a_box"] /
            items_slots_cfg["max_box_in_slot"]
        ).astype(int)
        items_slots_cfg["slot_weight"] = items_slots_cfg["item_weight"] * items_slots_cfg["max_item_in_slot"]

        # heaviest SKUs first so they get distributed across pods before lighter ones
        items_to_place = (
            items_slots_cfg
            .drop_duplicates("item_id")
            .sort_values(["item_class", "slot_weight"], ascending=[True, False])
            .copy()
        )

        # ----------------------------------------------------------------
        # 2. Build per-pod class slot registry
        #    pod_class_slots[pod_id][class] = [slot_id, ...] — available slots
        #    per class per pod, used for least-weight-first selection.
        # ----------------------------------------------------------------
        slots_per_pod = sum(class_slot_counts.values())

        # pod_class_slots: available (unassigned) slot_ids per pod per class
        pod_class_slots = {}
        for pod_id in pods["pod_id"].unique():
            pod_slots = pods.loc[
                (pods["pod_id"] == pod_id) & pods["item"].isnull(), "slot_id"
            ].tolist()
            if len(pod_slots) < slots_per_pod:
                continue
            pod_class_slots[pod_id] = {}
            cursor = 0
            for item_class, count in class_slot_counts.items():
                pod_class_slots[pod_id][item_class] = pod_slots[cursor: cursor + count]
                cursor += count

        # pod_weight: running total weight per pod (updated on each assignment)
        pod_weight = {
            pod_id: 0.0 for pod_id in pod_class_slots
        }

        # pod_class_counts: how many slots of each class have been assigned per pod
        pod_class_counts = {
            pod_id: {c: 0 for c in class_slot_counts}
            for pod_id in pods["pod_id"].unique()
        }

        def _assign_slot(pods, pod_id, slot_id, item_id, max_qty, item_weight, row):
            total_item_weight = round(item_weight * max_qty, 3)
            mask = (pods["pod_id"] == pod_id) & (pods["slot_id"] == slot_id)
            pods.loc[mask, "item"]                           = item_id
            pods.loc[mask, "qty"]                            = max_qty
            pods.loc[mask, "max_qty"]                        = max_qty
            pods.loc[mask, "item_weight"]                    = item_weight
            pods.loc[mask, "total_item_weight"]              = total_item_weight
            pods.loc[mask, "item_pod_inventory_level"]       = row["item_pod_inventory_level"]
            pods.loc[mask, "item_warehouse_inventory_level"] = row["item_warehouse_inventory_level"]
            pod_weight[pod_id] = round(pod_weight[pod_id] + total_item_weight, 3)

        def _pod_can_fit(pod_id, max_qty, item_weight):
            """Return True if adding a full slot stays within pod_wmax."""
            if item_weight <= 0:
                return True
            return (pod_weight[pod_id] + max_qty * item_weight) <= self.pod_wmax

        # ----------------------------------------------------------------
        # 3. Phase 1 — least-weight-first assignment with class-proportional slots
        #    For each SKU, sort candidate pods by current weight ascending,
        #    then pick the lightest pod that has a free class slot and fits weight.
        # ----------------------------------------------------------------
        partially_placed = {}
        needs_mopup = []

        for _, row in items_to_place.iterrows():
            item_id      = row["item_id"]
            item_class   = row["item_class"]
            item_weight  = row["item_weight"]
            max_qty      = int(items_slots_cfg.loc[
                items_slots_cfg["item_id"] == item_id, "max_item_in_slot"].values[0])
            slots_needed = int(items_slots_cfg.loc[
                items_slots_cfg["item_id"] == item_id, "slots_needed"].values[0])

            placed    = 0
            pods_used = set()  # max 1 slot per pod per SKU

            while placed < slots_needed:
                # Sort candidate pods by current weight (lightest first) each iteration
                candidates = sorted(
                    [pid for pid in pod_class_slots
                     if pid not in pods_used
                     and pod_class_slots[pid].get(item_class)  # has free class slot
                     and _pod_can_fit(pid, max_qty, item_weight)],
                    key=lambda pid: pod_weight[pid]
                )

                if not candidates:
                    # No pod with a class slot available — defer to mop-up
                    needs_mopup.append((item_id, item_class, item_weight, max_qty,
                                        slots_needed - placed, row))
                    break

                pod_id  = candidates[0]
                slot_id = pod_class_slots[pod_id][item_class].pop(0)
                if not pod_class_slots[pod_id][item_class]:
                    # Remove exhausted class entry so it won't appear as candidate
                    del pod_class_slots[pod_id][item_class]

                _assign_slot(pods, pod_id, slot_id, item_id, max_qty, item_weight, row)
                pods_used.add(pod_id)
                pod_class_counts[pod_id][item_class] += 1
                placed += 1

            partially_placed[item_id] = placed

        # ----------------------------------------------------------------
        # 4. Phase 2 — mop-up: assign remaining slots from any empty slot
        #    in any pod, as long as weight constraint is satisfied.
        #    Keeps mixed-class policy: pods already have class slots from phase 1.
        # ----------------------------------------------------------------
        if needs_mopup:
            print(f"    Mop-up phase: {len(needs_mopup)} items need additional slots")

        still_unallocated = []
        for item_id, item_class, item_weight, max_qty, slots_remaining, row in needs_mopup:
            placed    = 0
            pods_used = set(pods.loc[pods["item"] == item_id, "pod_id"].unique().tolist())
            limit     = class_slot_counts[item_class]

            while placed < slots_remaining:
                # Candidate pods: have any empty slot, not yet used by this SKU, fit weight
                candidate_pods = pods.loc[pods["item"].isnull(), "pod_id"].unique().tolist()

                found = False
                # first pass: respect class slot limit; second pass: relax if needed
                for strict in [True, False]:
                    if found:
                        break
                    # sort lightest first for weight balance in mop-up too
                    sorted_candidates = sorted(
                        [pid for pid in candidate_pods if pid not in pods_used],
                        key=lambda pid: pod_weight.get(pid, 0.0)
                    )
                    for pod_id in sorted_candidates:
                        if strict and pod_class_counts[pod_id][item_class] >= limit:
                            continue
                        if not _pod_can_fit(pod_id, max_qty, item_weight):
                            continue
                        empty_slots = pods.loc[
                            (pods["pod_id"] == pod_id) & pods["item"].isnull(), "slot_id"
                        ].tolist()
                        if not empty_slots:
                            continue
                        slot_id = empty_slots[0]
                        _assign_slot(pods, pod_id, slot_id, item_id, max_qty, item_weight, row)
                        pods_used.add(pod_id)
                        pod_class_counts[pod_id][item_class] += 1
                        placed += 1
                        found = True
                        break

                if not found:
                    print(f"    No pod available for item_id={item_id} "
                          f"(class {item_class}): placed {partially_placed.get(item_id,0)+placed}/{partially_placed.get(item_id,0)+slots_remaining} slots")
                    still_unallocated.append(item_id)
                    break

        # ----------------------------------------------------------------
        # 5. Finalize and save
        # ----------------------------------------------------------------
        pods["item"] = pods["item"].fillna(-1).astype(int)
        pods[["qty", "max_qty"]] = pods[["qty", "max_qty"]].fillna(0).astype(int)

        pods = pods.sort_values(["pod_id", "slot_sequence"]).reset_index(drop=True)
        pods["cumulative_pod_weight"] = (
            pods.groupby("pod_id")["total_item_weight"].cumsum().round(3)
        )

        assigned = pods[pods["item"] >= 0]
        pod_weights = pods.groupby("pod_id")["total_item_weight"].sum()
        print(f"\n=== Pod Assignment Summary ===")
        print(f"  SKUs assigned : {assigned['item'].nunique()} / {len(items_to_place)}")
        print(f"  Full slots    : {(assigned['qty'] == assigned['max_qty']).sum()}")
        print(f"  Partial slots : {(assigned['qty'] < assigned['max_qty']).sum()}")
        print(f"  Empty slots   : {(pods['item'] == -1).sum()}")
        print(f"  Pod weight — min: {pod_weights.min():.1f} kg  "
              f"mean: {pod_weights.mean():.1f} kg  max: {pod_weights.max():.1f} kg")
        if still_unallocated:
            print(f"  Unallocated item_ids: {still_unallocated}")
        print(f"==============================\n")

        pods.to_csv(working_path + "/pods.csv", index=False)
        return pods
    
    def config_items_pods(self,pod_types=[0], pod_num=[420], total_sku=500, 
                    #   items_class_conf={"A": 0.07, "B": 0.28, "C": 0.65},
                      items_class_conf={"A": 0.1, "B": 0.3, "C": 0.6},
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

