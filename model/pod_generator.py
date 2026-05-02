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

            # select items according to the class and its proportion
            items = pd.DataFrame()
            for class_name, conf in items_class_conf.items():

                # get list of the item's code based on the class
                item = item_df.loc[item_df["item_class"] == class_name].copy()
                item = item.sort_values(by="item_order_frequency", ascending=False)

                item["item_pod_inventory_level"] = items_pods_inventory_levels[class_name]
                item["item_warehouse_inventory_level"] = items_warehouse_inventory_levels[class_name]

                n_select = int(total_sku * conf)

                # select option 1: weighted sampling by 3-factor geometric importance score
                # else: top-N by order frequency
                if select_option == 1:
                    # Factor 1 — composite: mean_daily_demand × item_order_frequency
                    # Factor 2 — stability: item_order_frequency (proportional to 1/avg_demand_interval)
                    # Factor 3 — variability: cv²
                    f1 = item["mean_daily_demand"] * item["item_order_frequency"]
                    f2 = item["item_order_frequency"]
                    f3 = item["cv"] ** 2
                    importance = (f1.clip(lower=1e-9) * f2.clip(lower=1e-9) * f3.clip(lower=1e-9)) ** (1/3)
                    item_probability = (importance / importance.sum()).to_numpy()

                    item_code = np.random.choice(item["item_code"].to_list(), size=n_select,
                                                 p=item_probability, replace=False)
                    items = pd.concat([items, item.loc[item["item_code"].isin(item_code)]])
                else:
                    # select items based on the top n-percent of the class ratio
                    item_top = item.head(n_select)
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
                             class_slot_counts={"A": 13, "B": 4, "C": 3},
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

        pods["slot_sequence"] = np.arange(pods.shape[0])

        # ── Phase 2: Build virtual slot pools per class (round-robin across pods) ─
        # Round-robin ensures consecutive pool entries are from DIFFERENT pods,
        # so taking slots_needed consecutive entries places each slot in a different pod.
        print("    Building proportional slot pools (A=%d, B=%d, C=%d per pod)..." % (
            class_slot_counts.get("A", 0),
            class_slot_counts.get("B", 0),
            class_slot_counts.get("C", 0)))

        # Collect per-class slot indices grouped by pod, then interleave
        class_slots_by_pod = {cls: [] for cls in class_slot_counts}
        for _, grp in pods[pods["item"].isnull()].groupby("pod_id"):
            indices = grp.index.tolist()
            cursor = 0
            for cls, count in class_slot_counts.items():
                class_slots_by_pod[cls].append(indices[cursor:cursor + count])
                cursor += count

        # Interleave: [pod0_slot0, pod1_slot0, pod2_slot0, ..., pod0_slot1, pod1_slot1, ...]
        class_pools = {}
        for cls, pods_slots in class_slots_by_pod.items():
            max_slots = max(len(s) for s in pods_slots) if pods_slots else 0
            interleaved = []
            for slot_idx in range(max_slots):
                for pod_slots in pods_slots:
                    if slot_idx < len(pod_slots):
                        interleaved.append(pod_slots[slot_idx])
            class_pools[cls] = interleaved

        # ── Phase 3: Main assignment — class by class from pool ───────────────
        unplaced_ids = set()
        for cls, pool in class_pools.items():
            pool_cursor = 0
            cls_items = items_to_place[items_to_place["item_class"] == cls]
            print(f"    Class {cls}: {len(cls_items)} SKUs → {len(pool)} pool slots")

            for _, row in cls_items.iterrows():
                max_fit = int(row["max_item_in_slot"])
                if max_fit <= 0:
                    max_fit = 1
                slots_needed = int(np.ceil(row["item_initial_quantity_inventory"] / max_fit))
                remaining = len(pool) - pool_cursor

                if remaining >= slots_needed:
                    idxs = pool[pool_cursor:pool_cursor + slots_needed]
                    pods.loc[idxs, "item"]                          = int(row["item_id"])
                    pods.loc[idxs, "qty"]                           = max_fit
                    pods.loc[idxs, "max_qty"]                       = max_fit
                    pods.loc[idxs, "item_weight"]                   = row["item_weight"]
                    pods.loc[idxs, "total_item_weight"]             = round(row["item_weight"] * max_fit, 3)
                    pods.loc[idxs, "item_pod_inventory_level"]      = row["item_pod_inventory_level"]
                    pods.loc[idxs, "item_warehouse_inventory_level"]= row["item_warehouse_inventory_level"]
                    pool_cursor += slots_needed
                else:
                    unplaced_ids.add(int(row["item_id"]))

        # ── Phase 4: Mop-up — place overflow items in any remaining null slots ─
        # One slot per pod constraint: pick at most one empty slot per pod per item.
        if unplaced_ids:
            print(f"    Mop-up: {len(unplaced_ids)} items unplaced, using remaining empty slots...")
            mop_items = items_to_place[items_to_place["item_id"].isin(unplaced_ids)]
            for _, row in mop_items.iterrows():
                empty_df = pods[pods["item"].isnull()][["pod_id"]].copy()
                if empty_df.empty:
                    print(f"    WARNING: No slots left for item_id {int(row['item_id'])}")
                    break
                max_fit = int(row["max_item_in_slot"])
                if max_fit <= 0:
                    max_fit = 1
                slots_needed = int(np.ceil(row["item_initial_quantity_inventory"] / max_fit))
                # Pick one empty slot per pod, one pod per slot_needed
                # groupby keeps original pods index; take the first empty slot per pod
                one_per_pod_idx = empty_df.groupby("pod_id").apply(lambda g: g.index[0], include_groups=False).tolist()
                if len(one_per_pod_idx) >= slots_needed:
                    idxs = one_per_pod_idx[:slots_needed]
                else:
                    idxs = one_per_pod_idx
                if not idxs:
                    print(f"    WARNING: Not enough slots for item_id {int(row['item_id'])} (need {slots_needed}, have 0)")
                    continue
                pods.loc[idxs, "item"]                          = int(row["item_id"])
                pods.loc[idxs, "qty"]                           = max_fit
                pods.loc[idxs, "max_qty"]                       = max_fit
                pods.loc[idxs, "item_weight"]                   = row["item_weight"]
                pods.loc[idxs, "total_item_weight"]             = round(row["item_weight"] * max_fit, 3)
                pods.loc[idxs, "item_pod_inventory_level"]      = row["item_pod_inventory_level"]
                pods.loc[idxs, "item_warehouse_inventory_level"]= row["item_warehouse_inventory_level"]
                print(f"    Mop-up: item_id {int(row['item_id'])} (Class {row['item_class']}) placed in {len(idxs)} slots")

        # ── Phase 5: Finalize and save ────────────────────────────────────────
        pods[["item", "qty", "max_qty"]] = pods[["item", "qty", "max_qty"]].fillna(0).astype(int)
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

