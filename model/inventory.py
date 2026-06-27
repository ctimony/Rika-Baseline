from typing import Optional, List
import csv
import os
import math
import random
import threading
from collections import defaultdict, deque
import ast
import json
import pandas as pd
from datetime import datetime

from engine.landscape import Landscape
from engine.universe import Universe
from engine.util import *
from .intersection_manager import IntersectionManager
from .order import Order
from .order_manager import OrderManager
from .pod import Pod
from .pod_manager import PodManager
from .robot import Robot
from .robot_job import RobotJob
from .station_manager import StationManager
from .station import Station
from .storage_manager import StorageManager
from .storage import Storage
from .tools.write_record import write_record_to
# DB
from .tools.pod_location import get_pod_location
from .tools.order_history import upsert_order_history
from .tools.job_task import upsert_job_task, update_job_task
from .tools.pre_assign import initialize_pre_assign_table, clear_pre_assign_table, insert_pre_assign
# from .live_advanced_table import start_gui

# Show full column content
pd.set_option('display.max_colwidth', None)

# Show all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # Let it auto-expand

USE_OPPORTUNITY_SCORE = False     # True = Proactive ORS, False = baseline AND/OR gate with pod index
BASELINE_KL = 0.2              # Only used when USE_OPPORTUNITY_SCORE = False (AND baseline). KL 0.2 over the TRUE pod SKU count (n_p=20) → trigger at >=4/20 (20%) critical SKUs — the SAME trigger as the old capped form (ΣW_i/8 >= 0.4), now with the genuine fraction denominator.
BASELINE_UL = {'A': 0.6, 'B': 0.6, 'C': 0.6}  # Layer 1 utilization threshold per ABC class
# Baseline version (only used when USE_OPPORTUNITY_SCORE = False):
#   1 = AND/cascade with both layers required: a SKU must pass Layer 1 (flagged) AND
#       the pod must pass Layer 2 (Q_p >= KL). Needs the stockout override (Layer 1+2
#       rarely fire together). The original Chou/Yohana baseline.
#   2 = OR/cascade: Layer 1 fires alone — if any PICKED SKU has current_global/max_global
#       < UL_class, dispatch the pod immediately. Otherwise fall through to Layer 2
#       (Q_p >= KL). Either layer firing dispatches the pod. No stockout override needed
#       (Layer 1 catches depletion on its own).
BASELINE_VERSION = 1
# Stockout safety net for the AND baseline (BASELINE_VERSION == 1). When False,
# the AND baseline runs on its own logic ONLY (no override) — used to show the
# baseline stalls / under-replenishes without the net. When True, the net catches
# SKUs that hit zero stock as a last-resort floor.
STOCKOUT_OVERRIDE_ENABLED = True
# v13 stockout safety net (DEADLOCK GUARD). Separate from the baseline override above.
# When True and v13 is active, every scan interval we fetch ONE pod for each SKU whose
# WAREHOUSE stock is truly zero (current_global_qty == 0) and is still fillable — so a
# SKU that has gone to true global stockout (and whose pod never passes picking, e.g.
# when V13_ALLOW_FETCH is False) cannot leave orders hanging forever. It fires ONLY on
# true zero-stock (not on the opportunity score), identically across all weights, so it
# does not distort the weight comparison / Pareto front. Every firing is counted
# (self._v13_net_fetches) so the contamination can be quantified and reported.
V13_STOCKOUT_NET = False
# ORS version (only used when USE_OPPORTUNITY_SCORE = True):
#   6 = TWO-LEVEL ROP (pod-local): trigger over the just-picked SKUs; a SKU is
#       critical iff current_global <= rop_global AND slot_stock <= rop_per_slot.
#       If any is critical, replenish the just-picked pod directly. No candidate
#       search, no ranking. Cheapest mechanism — minimises trips (robot-bound).
#   7 = OPPORTUNITY SCORE + RPS (Replenishment Pod Selection, "check other pods"):
#       TRIGGER (same as v6 scope): a just-picked SKU is critical iff
#       current_global <= rop_global. For each critical SKU, gather CANDIDATE pods =
#       all pods carrying that SKU that have room in its slot (current_qty <
#       limit_qty) — the just-picked pod plus other pods (idle, in storage). Rank
#       candidates by opportunity score and replenish the best:
#         score(pod) = Σ_{s in pod} [ pending_order(s) × room(s) ]
#       where pending_order(s) = real-time committed demand (orders waiting now) and
#       room(s) = limit_qty - current_qty. The real-time pending term is the
#       contribution over Hsiao (2022), whose priority uses historical mean demand.
#       If the best pod is the just-picked one, it is dispatched for free (already in
#       hand); otherwise a robot retrieves the chosen pod from storage (RPS, a
#       standard RMFS decision). Selecting another pod only when its score is clearly
#       higher keeps the extra retrieval trips worthwhile in the robot-bound regime.
#  10 = REACTIVE after picking (Tracy-style, but with opportunity score):
#       TRIGGER: only just-picked SKUs checked against rop_global. If any newly
#       crosses ROP, evaluate ALL idle pods (including just-picked pod) that carry
#       any globally-critical SKU. Score: Σ pending(s) × (1 - current_pod(s)/limit_pod(s))
#       over all critical SKUs in the pod. No rop_per_slot gate. Pod with highest
#       score dispatched; if just-picked pod wins, it goes directly (no extra trip).
#   8 = CriticalDemandCoverage × TotalRefillGap over IDLE pods, event-driven fetch.
#   9 = opportunity-SCORED ranking (Σ pending·urgency) over IDLE pods, event-driven fetch.
ORS_VERSION = 20             # active version (see dispatch table below)
#  15 = v14 candidate set (pods sharing a picked-critical SKU) + WHOLE-POD fill +
#       WHOLE-POD reservation. Causal candidate restriction of v14, but each trip
#       refills/reserves the entire pod so the fill-rate sweep actually moves
#       energy/trips. v14 (critical-only) is left untouched as the comparator.
#  16 = v12 candidate set (EVERY idle pod carrying any critical SKU) + WHOLE-POD fill
#       + WHOLE-POD reservation. The whole-pod counterpart of v12 — completes the 2×2
#       (candidate: broad/narrow) × (fill: critical-only/whole-pod) factorial:
#       v12=broad+critical, v14=narrow+critical, v15=narrow+whole, v16=broad+whole.
#       v12 (critical-only) is left untouched as the comparator.
#  13 = weighted-sum opportunity score: Score = W_G·g(urgency) + W_E·emptiness
#       (event-driven, evaluated at finish-picking). The two weights trade off
#       service (anti-stockout) vs trip-efficiency; vary them for a Pareto front.
# LEAD_TIME_HOURS: lead time L (hours) used by the v16 stockout-probability score
# P_i = 1 − Φ((s − μ_h·L)/(σ_h·√L)). MUST equal ROP_HORIZON_SE_HOURS in
# 05_build_items_dictionary.py that produced the active rop_summary.csv — otherwise the
# score and the rop_global gate use different L and become inconsistent. Update together
# with rop_summary when sweeping L.
LEAD_TIME_HOURS = 1.0
V13_W_G = 0.0                # weight on g (urgency / anti-stockout)   — vary for Pareto
V13_W_E = 1.0                # weight on emptiness (trip-efficiency)   — V13_W_G + V13_W_E = 1
# FILL RATE: at the replenishment station, refill each TRIGGERING (critical) SKU up to
# V12_FILL_RATE × limit_qty (only those SKUs, not the whole pod). 1.0 = full; <1 leaves
# slots partly filled → lower pod mass → lower travel energy (prof's energy lever).
# Vary {0.7, 0.8, 1.0} as a DoE factor for the energy↔replenishment-frequency trade-off.
V12_FILL_RATE = 0.9
# WHOLE-POD FILL: when True, a replenishment trip refills EVERY SKU in the pod (up to
# V12_FILL_RATE × limit), not only the critical SKUs that triggered the trip. Since the
# pod is already at the replenishment station, topping up all its SKUs ('free-ride')
# raises pile-on and makes the pod stay full longer → fewer re-triggers / fewer trips,
# at the cost of higher pod mass (energy) and cross-SKU contamination of the per-SKU
# replenishment pattern. This is Kuo's proposed-policy behaviour (restock the whole
# selected pod to max). False = critical-only (cleaner per-SKU, more trips).
V12_FILL_WHOLE_POD = True
# Score_p = W_G·g_pod + W_E·emptiness_pod, where
#   g_pod        = MAX over the pod's CRITICAL SKUs of (rop_global−cur_global)/rop_global
#                  — how close the most-critical SKU is to a WAREHOUSE stockout (service).
#   emptiness_pod= 1 − (Σ current_qty / Σ limit_qty) over ALL SKUs in the pod — how empty
#                  the pod is overall (a replenish trip refills the WHOLE pod, so trip
#                  worth depends on the whole-pod deficit). Trip-efficiency.
# FETCH toggle: when V13_ALLOW_FETCH is False, a non-picked (idle) winner is NOT fetched
#   — it is left to be served the next time it passes through picking. This makes every
#   replenishment a PIGGYBACK (zero fetch trips) across ALL weights, so the trip count
#   differs only by which picked pods get piggybacked → a clean weight comparison.
V13_ALLOW_FETCH = True
# v8/v9 replenishment is EVENT-DRIVEN (decided when a free robot asks for a task,
# the WES robot life-cycle decision point of Bolu & Korçak 2021) — no periodic scan
# cadence. A pod is dispatched only if it carries SKUs critical at BOTH the global
# (current_global ≤ rop_global, net of in-flight reservations) AND per-pod
# (slot_stock ≤ rop_per_slot) levels; in-flight reservation prevents a second pod
# being sent for a SKU already covered by an en-route pod.
# Experimental extreme lower-bound point (v6): when True, the per-slot trigger fires
# ONLY when the slot is fully depleted (slot_stock == 0) — i.e. ROP_per_slot = 0,
# no safety stock, reorder only after stockout. Used to map the bottom of the L/ROP
# trade-off curve (proves a positive ROP / ~30 min lead time is better). NOT a
# realistic operating policy — a theoretical worst-case reference.
TRIGGER_ON_EMPTY_ONLY = False
class Inventory(Universe):
    dimension = 60
    map = []
    landscape = None
    stop_and_go = 0
    total_energy = 0
    total_pod = 0
    total_turning = 0
    total_robot_idle = 0
    movement_channel = {}
    graph = None
    graph_pod = None

    def __init__(self):
        self._tick = 0  #current counter
        # self.ignored_types = ["pod", "station", "way-direction"]
        self.ignored_types = ["station", "way-direction"]
        self.tick_to_second = 0.15
        self.job_queue: list[RobotJob] = []
        self.landscape = Landscape(self.dimension)
        self.pod_manager = PodManager()
        self.station_manager = StationManager()
        self.storage_manager = StorageManager(self)
        self.order_manager = OrderManager()
        self.next_process_tick = 0
        self.intersection_manager = IntersectionManager(self.landscape.current_date_string)
        self.update_intersection_using_RL = False
        self.zoning = False
        self.robot_queue_order = {}
        self.preassign_dict = {}
        self.last_order = {}
        
        self.preassign_per_station = defaultdict(deque)
        # v7 anti-redundancy: per-SKU units already PROMISED by pods currently in
        # transit to a replenishment station. A SKU's effective global stock is
        # current_global + reserved_global, so a second pod is not dispatched for
        # a SKU that an in-flight pod will already restock above its ROP.
        self.reserved_global = defaultdict(float)
        # self.currently_picking = {}
        # # Shared wrapper for the DataFrame
        # self.shared_data = {"df": pd.DataFrame()}

        # # Start GUI in a thread
        # self.gui_thread = threading.Thread(target=start_gui, args=(self.shared_data,), daemon=True)
        # self.gui_thread.start()
        self.poa_podmatch = False
        self.poa_first = False  # preasign2 gajelas nih / F3
        self.poa_second = True

        self.pps_pileon = True
        self.pps_demand = False

        self.priority_order = False
        self._stockout_skus: set = set()  # rebuilt every tick
        self._v13_net_fetches = 0  # count of v13 stockout-net (deadlock-guard) fetches
        self.skus_replenished_count = 0  # total SKU slots refilled across all replenishment trips
        # Diagnostic: why does an order wait? (bottleneck attribution)
        self._wait_stockout = 0  # times PPS found no pod that can contribute (SKU depleted)
        self._wait_robot = 0     # times a pod was found but no idle robot to carry it
        # Stockout Count (performance metric): UNIQUE (order_id, sku) order lines that
        # encounter a true stockout — warehouse stock of the requested SKU is zero
        # (i.e. all pods storing it are empty, per Lamballais et al. 2018) at the moment
        # the order line is processed. Counted once per order-line (a set), unlike
        # _wait_stockout which accumulates every processing tick.
        self._stockout_orderlines = set()

        if self.poa_second:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            initialize_pre_assign_table(timestamp)
            clear_pre_assign_table()
        super().__init__()

    def addObject(self, object):
        if object.object_type == "robot":
            object._id = self.total_pod + 1
            self.total_pod += 1
        super().addObject(object)

    def addTrafficPolicyHistory(self, sender, target):
        if target not in self.movement_channel:
            self.movement_channel[target] = []
        self.movement_channel[target].append(sender)

    def getTrafficPolicyHistory(self, target):
        if target not in self.movement_channel:
            return []
        return self.movement_channel[target]

    def tick(self):
        # Get initial state
        result = super().generateResult()
        
        print(f"Current tick: {self._tick}")

        # Reset movement tracking
        self.movement_channel = {}

        # Stockout safety net (AND baseline only) — scan every 3600 s for SKUs that have
        # ACTUALLY hit zero global stock and dispatch a REAL robot trip (a pod carrying the
        # stocked-out SKU is fetched to a replenishment station, refilling ONLY the critical
        # stocked-out SKU on arrival). This is a robot-travel trip COUNTED in the trip/energy
        # metrics — the same cost basis as the v13 fetch that handles stockouts on the
        # opportunity-score side — so baseline and ORS pay the same price for a stockout
        # recovery, keeping the comparison fair.
        if STOCKOUT_OVERRIDE_ENABLED and not USE_OPPORTUNITY_SCORE and BASELINE_VERSION == 1:
            if int(self._tick) % 3600 == 0:
                self._stockout_skus = {
                    sku for sku, data in self.pod_manager.skus_data.items()
                    if data.get('current_global_qty', 1) == 0
                }
            if self._stockout_skus:
                self._dispatch_stockout_pods()        # robot travel (real trip), not instant

        # v13 stockout safety net (deadlock guard) — same scan as the baseline net, but
        # gated on v13 being active. Fires ONLY for true zero-stock SKUs, identically
        # across all weights, so it does not distort the weight comparison.
        if V13_STOCKOUT_NET and USE_OPPORTUNITY_SCORE and ORS_VERSION in (12, 13, 14, 15, 16, 17, 19, 20):
            if int(self._tick) % 3600 == 0:
                self._stockout_skus = {
                    sku for sku, data in self.pod_manager.skus_data.items()
                    if data.get('current_global_qty', 1) == 0
                }
            if self._stockout_skus:
                self._dispatch_stockout_pods()        # robot travel (real trip) — SAME cost
                                                       # basis as the baseline stockout override,
                                                       # so v12/v13 and baseline are fair.

        # Process orders at scheduled intervals
        if int(self._tick) == self.next_process_tick:
            print(f"Processing orders at tick {self._tick}")
            self.find_new_orders()
            self.process_orders()
            if self.update_intersection_using_RL:
                self.intersection_manager.update_allowed_direction_using_q_model(int(self._tick))
        print(f"Current job queue length: {len(self.job_queue)}")

        # v8/v9 REPLENISHMENT — EVENT-DRIVEN (after the WES robot life-cycle of Bolu &
        # Korçak 2021): replenishment is decided at the moment a robot is free and asks
        # for a task — the standard WES decision point — NOT by periodic scanning. This
        # avoids imposing a separate continuous decision loop on the warehouse execution
        # system while staying responsive to SKU depletion. When a robot is free, if a
        # pod carrying two-level-critical SKUs (needed by waiting orders) exists, the
        # robot is sent to replenish the best such pod instead of picking; in-flight
        # reservation prevents two pods being sent for the same SKU.
        #   v9 = opportunity-SCORED ranking (Σ pending·urgency), fetch only;
        #   v8 = CriticalDemandCoverage × TotalRefillGap over IDLE pods only,
        #        dispatches all worthwhile pods this tick (each fetched by a robot);
        #   v7 = PIGGYBACK (finish_picking, no fetch) + FETCH for the rest — both use
        #        the v9 opportunity score; the per-tick scan below is v7's FETCH half.
        # v12 is EVENT-DRIVEN (decided at finish-picking in finish_picking_task), so there
        # is NO periodic scan here — it evaluates only when a pod finishes picking.
        if USE_OPPORTUNITY_SCORE and ORS_VERSION in (8, 9):
            self._dispatch_replenishment_event(use_score=(ORS_VERSION == 9))
        elif USE_OPPORTUNITY_SCORE and ORS_VERSION == 10:
            # v10 = continuous-review SCAN-FETCH every tick: build the ROP-global critical
            # set, rank ALL idle pods by the v10 opportunity score (Σ g_i·(1−cur/lim)·pending)
            # and greedily fetch worthwhile pods (bounded by free robots / station capacity).
            # No piggyback, no stockout override.
            # (v11 and v12 do NOT scan per tick — they evaluate ONLY at finish-picking;
            #  v11 = greedy multi-dispatch, v12 = one pod per picking event.)
            self._dispatch_replenishment_event(use_score=True)
        elif USE_OPPORTUNITY_SCORE and ORS_VERSION == 7:
            # v7 = PIGGYBACK (decided at finish-picking, no fetch) + FETCH for the rest.
            # The piggyback in finish_picking_task serves critical pods that happen to
            # pass through picking for free; this event-driven scan fetches the best
            # IDLE pod for critical+awaited SKUs that did NOT pass through picking. Both
            # use the SAME opportunity score (_v9_pod_score via use_score=True), and the
            # in-flight reservation set is shared, so a SKU already covered by a piggyback
            # pod is not fetched again. Dispatch only consumes a free robot, so picking is
            # not starved.
            self._dispatch_replenishment_event(use_score=True)

        if len(self.job_queue) > 0:
            job = self.job_queue[0]

            if job is not None:
                current_distance = float("inf")
                nearest_robot: Optional[Robot] = None

                for o in self.get_movable_objects():
                    if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                        dist = calculateDistance(o.pos_x, o.pos_y, job.pod_coordinate.x, job.pod_coordinate.y)
                        if dist < current_distance:
                            nearest_robot = o
                            current_distance = dist

                # A queued replenishment fetch may have been queued without a station slot
                # (station was full at enqueue time). Try to bind a station now; if still
                # none, leave it queued for a later tick.
                if nearest_robot is not None and getattr(job, 'station_id', None) is None and job.orders == []:
                    st = self.station_manager.find_available_replenish_station()
                    if st is not None:
                        st.add_pod(job.pod.pod_id)
                        job.station_id = st.station_id
                    else:
                        nearest_robot = None   # keep waiting until a station frees up

                if nearest_robot is not None:
                    self.job_queue.remove(job)  # Remove the selected job from the queue
                    print(f"Assigning job {job.pod}-{job.station_id} to robot {nearest_robot._id}")
                    nearest_robot.assign_job_and_set_move_to_take_pod(job)
                else:
                    # A job is queued (pod+SKUs ready) but no idle robot to carry it
                    # (robot-induced wait).
                    self._wait_robot += 1
                    for triplet in job.orders:
                        upsert_job_task(
                            pod_id=str(job.pod.pod_id),
                            order_id=str(triplet[0]),
                            sku=str(triplet[1]),
                            qty=str(triplet[2]),
                            status="otw",
                        )
            

        # Update object positions and collect metrics
        total_energy = 0
        total_turning = 0
        total_idle = 0
        tick_load_mass = 0.0      # total pod mass carried by MOVING robots this tick
        tick_moving_robots = 0    # how many robots are actually moving this tick
        for o in self.get_movable_objects():
            if isinstance(o, Robot):
                initial_velocity = o.velocity
                o.move()
                total_energy += o.energy_consumption
                total_turning += o.turning
                total_idle += (o.total_idle * 0.15)
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1
                # Diagnostic: mass carried while moving (energy ∝ (mass+load_mass)·v)
                if o.velocity != 0:
                    tick_load_mass += getattr(o, 'load_mass', 0.0)
                    tick_moving_robots += 1

                # Handle job completion and replenishment
                if o.job is not None and o.job.picking_delay == 0 and not o.job.is_finished:
                    need_replenish_pod = self.finish_task_in_job(o.job, robot=o)
                    for triplet in o.job.orders:
                        update_job_task(
                            pod_id=str(o.job.pod.pod_id),
                            order_id=str(triplet[0]),
                            sku=str(triplet[1]),
                            qty=str(triplet[2]),
                            status="finish",
                            finish_time=self._tick
                        )
                    if need_replenish_pod:
                        # pod: Pod = self.pod_manager.get_pod_by_coordinate(o.job.pod_coordinate.x, o.job.pod_coordinate.y)
                        pod: Pod = self.pod_manager.get_pod_by_id(o.job.pod.pod_id)
                        latest_pod_location = get_pod_location(pod.pod_id)
                        if latest_pod_location:
                            pod.pos_x, pod.pos_y = int(latest_pod_location[0]), int(latest_pod_location[1])
                        station_replenish = self.station_manager.find_available_replenish_station()
                        if station_replenish is not None:
                            station_replenish.add_pod(pod.pod_id)
                            new_job = RobotJob(pod.coordinate, station_id=station_replenish.station_id, pod=pod)
                            new_job.add_replenishment_task(pod)
                            o.assign_job_and_set_move_to_station(new_job)
                        else:
                            # No replenishment station slot free — pod returns to
                            # idle, replenishment re-triggered on next pick. Release
                            # any v7 reservation so it does not leak (the promised
                            # refill will NOT happen this trip).
                            reserved = getattr(pod, 'reserved_fill', None)
                            if reserved:
                                for sku, fill in reserved.items():
                                    self.reserved_global[sku] = max(0.0, self.reserved_global.get(sku, 0.0) - fill)
                                pod.reserved_fill = {}
                    # If deferred (need_replenish_pod=False), robot continues returning_pod normally.
                    # mark_pod_available is called at line below when robot reaches idle state.

                # Reset completed jobs
                if o.current_state == 'idle' and o.job is not None:
                    # self.pod_manager.mark_pod_available(o.job.pod_coordinate)
                    self.pod_manager.mark_pod_available(o.job.pod)
                    o.job = None
                
                # Modify job if a new order is assign while pod is on the way
                # TODO:
                if o.job is not None:
                    self.update_robot_job_for_new_orders(o.job)

        # Update global metrics
        self.total_robot_idle = total_idle
        self.total_energy = total_energy / 1000  # TEC in kJ (Eq. 3.30); per-robot energy stays in Joule
        self.total_turning = total_turning
        self._tick_load_mass = tick_load_mass          # diagnostic: mass carried while moving
        self._tick_moving_robots = tick_moving_robots

        # Update process tick and intersection model
        if int(self._tick) == self.next_process_tick:
            self.next_process_tick += 1
            if self.update_intersection_using_RL:
                self.intersection_manager.update_model_after_execution(self._tick)

        # Increment tick
        self._tick += self.tick_to_second

        # Return updated state with station orders
        station_orders = self.get_station_orders_info()
        # with open('result.txt', 'a') as f:
        #     f.write(f"{result}")
        return [result, station_orders]

    def finish_task_in_job(self, job: RobotJob, robot=None):
        job_station = self.station_manager.get_station_by_id(job.station_id)
        if job_station.is_picker_station():
            try:
                return self.finish_picking_task(job, robot=robot)
            except Exception as e:
                print(f"[ERROR] finish_picking_task for job {job.job_id}")
                print(f"[ERROR] for pod {job.pod} location {job.pod.coordinate}")
                raise e
        elif job_station.is_replenishment_station():
            try:
                return self.finish_replenishment_task(job)
            except Exception as e:
                print(f"[ERROR] finish_replenishment_task for job {job.job_id}")
                print(f"[ERROR] for pod {job.pod} location {job.pod.coordinate}")
                raise e
    
    def finish_picking_task(self, job: RobotJob, robot=None):
        # pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod: Pod = self.pod_manager.get_pod_by_id(job.pod.pod_id)
        # APPEND-ONLY rows for pod_info: collect this job's picking rows and append at
        # the end (mode='a', no header). Do NOT read+concat+overwrite the whole file —
        # picking fires thousands of times and replenishment dozens; the old
        # read-modify-write race let a picking write overwrite (drop) replenishment
        # rows committed by another event, undercounting total_replenishments.
        picking_rows = []
        for order_id, sku, quantity in job.orders:
            order: Order = self.order_manager.get_order_by_id(order_id)
            order.deliver_quantity(sku, quantity)
            print("order, sku, quantity :" ,order_id, sku, quantity)

            assign_order_df = pd.read_csv('assign_order.csv')
            assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'status'] = 1
            assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'order_finished'] = int(self._tick)
            assign_order_df.to_csv('assign_order.csv', index=False)
            picking_rows.append({
                "pod_id": pod.pod_id,
                "item_id": sku,
                "qty": quantity,
                "order_id": order_id,
                "processed_time": int(self._tick),
                "task_type": 1,
                "trigger": None,
                "opp_score": None,
            })

            if order.is_order_completed():
                self.order_manager.finish_order(order_id, int(self._tick))
                station = self.station_manager.get_station_by_id(order.station_id)
                station.remove_order(order_id,order)
                self.insert_finished_order_to_csv(order)
                # DB
                # if not isinstance(order.order_id, int):
                    # raise AssertionError(f"WHAT? order {order} order_id {order.order_id} order_id {order_id}")
                upsert_order_history(order_id, order_finish_time=self._tick)
        station = self.station_manager.get_station_by_id(job.station_id)
        station.remove_pod(pod.pod_id)

        # Check only recently picked SKUs against current global ROP state
        sku_need_replenished = []
        for _, sku, _ in job.orders:
            sku_out, replenished_status = self.pod_manager.is_sku_need_replenished(sku)
            if replenished_status and sku_out not in sku_need_replenished:
                sku_need_replenished.append(sku_out)

        if picking_rows:
            pd.DataFrame(picking_rows, columns=["pod_id", "item_id", "qty", "order_id",
                "processed_time", "task_type", "trigger", "opp_score"]).to_csv(
                'pod_info.csv', mode='a', header=False, index=False)
        job.set_job_finish()
        if USE_OPPORTUNITY_SCORE:
            # v8/v9/v10 decide replenishment in the main loop (scanned every tick); the
            # just-picked pod simply returns to storage (no piggyback). v10 ranks pods with
            # the v10 opportunity score (Σ g_i·(1−cur/lim)·pending); this isolates the
            # SCORE from any piggyback mechanism.
            if ORS_VERSION in (8, 9, 10):
                return False
            # v11 = v10 + PIGGYBACK with RANKED-TOGETHER selection. The just-picked pod is
            # already in the robot's hands; score it TOGETHER with all idle pods (same v10
            # score). If the just-picked pod wins, it goes straight to replenishment for
            # free (no fetch trip); if an idle pod wins, that pod is fetched and the
            # just-picked pod returns to storage. Reservation is shared with the per-tick
            # scan so no SKU is double-served. Pods not passing through picking are served
            # by that scan.
            if ORS_VERSION == 11:
                picked_skus = {sku for _, sku, _ in job.orders}
                return self._dispatch_piggyback_ranked_v11(picked_pod=pod,
                                                           picking_robot=robot,
                                                           picked_skus=picked_skus)
            # v12 = EVENT-DRIVEN at finish-picking: rank the just-picked pod (and idle pods)
            # by the multiplicative opportunity score Σ g·(1−cur/lim). If the just-picked
            # pod wins → piggyback (no fetch). Fetch is disabled (V13_ALLOW_FETCH=False), so
            # idle winners are left for the next pick; the stockout override (robot travel)
            # catches true zero-stock SKUs.
            # v16 = v12's broad candidate set (every idle pod carrying any critical SKU)
            # but WHOLE-POD fill + WHOLE-POD reservation (fill/reserve controlled below by
            # ORS_VERSION). Uses v12's dispatch; v12 (critical-only) stays untouched.
            if ORS_VERSION in (12, 16, 17):
                picked_skus = {sku for _, sku, _ in job.orders}
                return self._dispatch_piggyback_ranked_v12(picked_pod=pod,
                                                           picking_robot=robot,
                                                           picked_skus=picked_skus)
            # v13 = v12 event-driven piggyback+fetch, but ranks pods with the
            # weighted-sum score (W_G·urgency + W_E·emptiness) instead of v12's
            # multiplicative Σ g·(1−cur/lim).
            if ORS_VERSION == 13:
                picked_skus = {sku for _, sku, _ in job.orders}
                return self._dispatch_piggyback_ranked_v13(picked_pod=pod,
                                                           picking_robot=robot,
                                                           picked_skus=picked_skus)
            # v14 = v12 EXACTLY (same Σ g·(1−cur/lim) score, same reserve, same fill-rate,
            # same piggyback+fetch), but the candidate set is RESTRICTED to pods that share
            # a critical SKU with the just-picked pod (not every pod carrying any critical
            # SKU). Causally cleaner: picking pod X → look only at pods that also carry X's
            # critical SKUs, so a fetch is always relevant to the SKU that just went critical.
            # v15 = v14's candidate set (pods sharing a picked-critical SKU) but WHOLE-POD
            # fill + WHOLE-POD reservation (every SKU in the dispatched pod is refilled and
            # reserved, not only the critical ones). Lets the fill-rate sweep actually move
            # energy/trips (fill rate now multiplies ~all slots, not ~1-2), while keeping
            # v14's causal candidate restriction. v14 (critical-only) stays untouched.
            # v19 = v14/v15's causal narrow candidate set + whole-pod fill, but ranks
            # candidates with the Hsiao opportunity-cost score _v19_pod_score
            # (Σ max(0,rop−cur)·(lim−cur) over ALL SKUs, no divisor) instead of v15's
            # Σ g·(1−cur/lim). Routes through the v14 dispatch; the score branch inside
            # picks _v19_pod_score when ORS_VERSION == 19. v14/v15 stay untouched.
            if ORS_VERSION in (14, 15, 19, 20):
                picked_skus = {sku for _, sku, _ in job.orders}
                return self._dispatch_piggyback_ranked_v14(picked_pod=pod,
                                                           picking_robot=robot,
                                                           picked_skus=picked_skus)
            # v7 = PIGGYBACK + opportunity score. The just-picked pod is already in the
            # robot's hands at the station, so if it carries SKUs that are critical AND
            # awaited by orders, sending it straight to replenishment costs NO fetch trip
            # (the robot continues instead of returning empty). The decision is taken
            # HERE, at finish-picking — the pod is already free, so nothing is held back
            # (this avoids the "pod tied up" failure of the old v7, which reserved pods
            # mid-pick). Pods NOT passing through picking are handled by the event-driven
            # fetch scan (_dispatch_replenishment_event) in the main loop.
            picked_skus = {sku for _, sku, _ in job.orders}
            return self._dispatch_proactive_replenishment(extra_pod=pod, picked_skus=picked_skus)

        # ---- Baseline (Warehouse Inventory–SKU in Pod) ----
        # Layer 1 — flag SKUs in the pod whose warehouse utilization current_global /
        # max_global is below the per-class threshold UL.
        flagged = []
        for sku in pod.skus:
            data = self.pod_manager.skus_data.get(sku)
            if data is None:
                continue
            max_global = data.get('max_global_qty', 0)
            if max_global <= 0:
                continue
            current_global = data.get('current_global_qty', 0)
            item_class = data.get('item_class', 'C')
            ul = BASELINE_UL.get(item_class, BASELINE_UL['C'])
            if current_global / max_global < ul:
                flagged.append(sku)

        if BASELINE_VERSION == 2:
            # OR / cascade: Layer 1 fires on its own. If any PICKED SKU is flagged
            # (below UL), dispatch immediately (trigger='global'). Otherwise fall
            # through to Layer 2 (trigger='pod').
            picked_skus = {sku for _, sku, _ in job.orders}
            if any(sku in picked_skus for sku in flagged):
                pod.last_trigger = 'global'   # Layer 1 fired
                # Refill ONLY the flagged (critical) SKUs (flagged-only, like V1).
                pod.reserved_fill = {sku: 0.0 for sku in flagged if sku in pod.skus}
                return True
            if flagged and pod.check_pod_index(flagged, BASELINE_KL):
                pod.last_trigger = 'pod'       # Layer 2 fired
                # Refill ONLY the flagged (critical) SKUs (flagged-only, like V1).
                pod.reserved_fill = {sku: 0.0 for sku in flagged if sku in pod.skus}
                return True
            return False

        # BASELINE_VERSION == 1 — AND: both layers required.
        if len(flagged) == 0:
            return False
        # Layer 2 — Q_p = Σ W_i(flagged) / n_p >= KL  (W_i binary, à la Tracy/Hsiao)
        # denom = n_p (true pod SKU count = 20); Q_p is the genuine critical fraction.
        if pod.check_pod_index(flagged, BASELINE_KL):
            pod.last_trigger = 'global'
            # Refill ONLY the flagged (critical) SKUs — same critical-only rule as the
            # opportunity-score policies (v12/v13), so the comparison is fair. flagged =
            # the SKUs below the UL threshold that triggered this pod. reserved_fill drives
            # the critical-only refill in finish_replenishment_task.
            pod.reserved_fill = {sku: 0.0 for sku in flagged if sku in pod.skus}
            return True
        return False

    def finish_replenishment_task(self, job: RobotJob):
        # pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod: Pod = self.pod_manager.get_pod_by_id(job.pod.pod_id)
        qty_before = {sku: d['current_qty'] for sku, d in pod.skus.items()}
        # Refill ONLY the CRITICAL SKUs that triggered this trip (the reserved set),
        # each up to V12_FILL_RATE × limit_qty — NOT the whole pod. Filling the whole
        # pod 'free-rides' non-critical SKUs (they get restocked without ever being
        # critical), which (a) cross-contaminates each SKU's replenishment pattern and
        # (b) maximises pod mass → travel energy. Filling only the triggering SKUs at a
        # chosen fill rate keeps replenishment per-SKU clean and controls pod mass
        # (fill rate = prof's energy lever). Falls back to whole-pod only if no reserved
        # set is recorded (safety).
        reserved = getattr(pod, 'reserved_fill', None)
        stockout_fill = getattr(pod, 'stockout_fill_skus', None)
        # REFILL SCOPE is tied to the POLICY, so v12 and v13 differ not only in their
        # scoring formula but in what a trip actually fills:
        #   • v13 (and any policy with V12_FILL_WHOLE_POD) = WHOLE-POD fill: top up EVERY
        #     SKU up to the fill rate (free-ride), matching v13's whole-pod emptiness
        #     score → score and refill are consistent (both whole-pod).
        #   • v12 = CRITICAL-ONLY fill: refill ONLY the critical SKUs that scored the trip
        #     (reserved_fill = the covered critical set), matching v12's critical-only
        #     score Σ g_i·(1−cur/lim) → score and refill are consistent (both critical).
        #     No free-ride of non-critical SKUs; cleaner per-SKU pattern, lower pod mass.
        # WHOLE-POD fill applies to:
        #   • v13 (and any opportunity-score policy with V12_FILL_WHOLE_POD), and
        #   • the BASELINE's GLOBAL (proactive) trigger — the "Warehouse inventory-SKU in
        #     pod" policy replenishes ALL SKUs in the pod once a pod is sent (Hsiao 2022,
        #     "replenish all SKUs in this pod"). The baseline's STOCKOUT override is the
        #     reactive exception below (stockout_fill) — it tops up ONLY the stocked-out
        #     critical SKU, not the whole pod.
        # v12 and v14 are the critical-only exceptions (whole_pod_fill stays False for them).
        # v15 = v14 candidate set but WHOLE-POD fill, so it is NOT excluded here: v15 always
        # fills the whole pod (its defining behaviour), independent of V12_FILL_WHOLE_POD.
        if not USE_OPPORTUNITY_SCORE:
            # BASELINE: global (proactive) trigger refills ONLY the FLAGGED (critical)
            # SKUs that triggered the pod (reserved_fill), each up to V12_FILL_RATE ×
            # limit_qty — NOT the whole pod. This matches the proposed policy's flagged-
            # only refill so the comparison is isolated to the pod-selection rule, and it
            # avoids free-riding non-critical SKUs (cleaner per-SKU pattern, lower pod
            # mass). The stockout override still fills ONLY the stocked-out SKU.
            whole_pod_fill = False
        elif ORS_VERSION in (15, 16, 19, 20):
            # v15/v16/v19 ALWAYS whole-pod (their whole point); stockout override still
            # critical-only.
            whole_pod_fill = not stockout_fill
        else:
            # Other opportunity-score policies: v13 (and any with V12_FILL_WHOLE_POD) =
            # whole-pod; v12, v14 and v17 = critical-only (v17 = v12 + g², so it must match
            # v12's fill scope — differ ONLY in the urgency exponent, nothing else).
            whole_pod_fill = V12_FILL_WHOLE_POD and ORS_VERSION not in (12, 14, 17)
        if whole_pod_fill:
            # WHOLE-POD PER-SLOT FILL (v15/v16/v19 and any V12_FILL_WHOLE_POD policy):
            # top up EVERY SKU to V12_FILL_RATE × its own slot capacity (limit_qty). The
            # fill rate is the energy lever — filling each slot to φ·limit (not the whole
            # pod's mass budget) keeps pod mass = φ·capacity, so trips rise LINEARLY as φ
            # falls (predictable), unlike the pod-capacity scheme which left slots filled
            # unevenly and made trips explode. Matches the thesis fill rule z_ip ≤ φ·C_i.
            pod.replenish_skus_fillrate(list(pod.skus.keys()), fill_rate=V12_FILL_RATE)
            pod.stockout_fill_skus = set()
        elif stockout_fill:
            # Stockout-triggered robot trip (baseline _dispatch_stockout_pods): refill ONLY
            # the stocked-out (critical) SKU(s) that triggered this trip, at the same fill
            # rate — NOT the whole pod. Keeps trip mass/energy comparable to proactive trips.
            pod.replenish_skus_fillrate(list(stockout_fill), fill_rate=V12_FILL_RATE)
            pod.stockout_fill_skus = set()             # clear marker (consumed)
        elif reserved:
            pod.replenish_skus_fillrate(list(reserved.keys()), fill_rate=V12_FILL_RATE)
        else:
            pod.replenish_all_skus()                   # safety fallback (no reserved set)
        skus_filled_this_trip = 0
        for sku, d in pod.skus.items():
            qty_added = d['current_qty'] - qty_before.get(sku, 0)
            if qty_added > 0:
                self.pod_manager.restore_sku_data(sku, qty_added)
                skus_filled_this_trip += 1
        self.skus_replenished_count += skus_filled_this_trip
        # Release the reservation this pod held: its refill is now reflected in
        # the real current_global (restore_sku_data above), so the in-flight
        # promise is no longer needed. Release exactly what was reserved. (`reserved`
        # already read above for the critical-SKU refill.)
        # Log which SKUs (and their class) TRIGGERED this replenishment — the
        # reserved set = the critical SKUs this pod was dispatched to cover. This
        # tells us, per trip, which class actually drives replenishment.
        if reserved:
            with open('replenish_trigger_log.csv', 'a') as _f:
                for sku in reserved:
                    d = self.pod_manager.skus_data.get(sku, {})
                    cls = d.get('item_class', '?')
                    _f.write(f"{int(self._tick)},{pod.pod_id},{sku},{cls},"
                             f"{pod.last_trigger}\n")
        if reserved:
            for sku, fill in reserved.items():
                self.reserved_global[sku] = max(0.0, self.reserved_global.get(sku, 0.0) - fill)
            pod.reserved_fill = {}
        # APPEND-ONLY (mode='a', no header) — never read+concat+overwrite the whole
        # file, so a concurrent picking write cannot drop this replenishment row.
        trip_start = getattr(job, 'start_tick', None)
        trip_duration = (int(self._tick) - int(trip_start)) if trip_start is not None else ""
        new_row = {
                "pod_id": pod.pod_id,
                "item_id": -1,
                "qty": -1,
                "order_id": -999,
                "processed_time": int(self._tick),
                "task_type": 2,
                "trigger": pod.last_trigger,
                "opp_score": pod.last_opp_score,
                "trip_start": int(trip_start) if trip_start is not None else "",
                "trip_duration": trip_duration
            }
        pd.DataFrame([new_row], columns=["pod_id", "item_id", "qty", "order_id",
            "processed_time", "task_type", "trigger", "opp_score",
            "trip_start", "trip_duration"]).to_csv(
            'pod_info.csv', mode='a', header=False, index=False)
        # job.is_finished = True
        job.set_job_finish()
        station = self.station_manager.get_station_by_id(job.station_id)
        station.remove_pod(pod.pod_id)
        return False

    def _dispatch_stockout_instant(self):
        """INSTANT stockout safety net: for each truly stocked-out SKU (current_global==0),
        refill ONLY that stocked-out SKU in its pod immediately so the waiting orders can
        proceed. This prevents deadlock in the event-driven system (a stocked-out SKU's pod
        would otherwise never be dispatched, since its orders cannot complete). The refill is
        EMERGENCY fulfilment OUTSIDE the regular robot replenishment operation, following
        Aerts et al. (2022) and Zhang & Chen (2022) who model stockout/emergency as a SEPARATE
        metric, not a replenishment trip. Therefore it does NOT write a replenishment-trip row,
        does NOT add to skus_replenished, and does NOT consume robot/energy — it is recorded
        only via the emergency-net counter (reported alongside the stockout count). This keeps
        the replenishment-trip and energy metrics reflecting the replenishment policy alone."""
        # Group by pod so one pod refilling several stocked-out SKUs is one emergency event.
        skus_to_remove = set()
        pod_fills = {}  # pod_id -> (pod_obj, [sku_id, ...])
        for sku_id in list(self._stockout_skus):
            pods_for_sku = self.pod_manager.sku_to_pods.get(sku_id) or []
            chosen = None
            for pod in pods_for_sku:
                if pod is None or sku_id not in pod.skus:
                    continue
                d = pod.skus[sku_id]
                if d['current_qty'] >= d['limit_qty']:
                    continue
                chosen = pod
                break
            skus_to_remove.add(sku_id)            # handled or unfillable — drop from set
            if chosen is not None:
                pod_fills.setdefault(chosen.pod_id, (chosen, []))[1].append(sku_id)

        # Emergency refill: ONLY the stocked-out SKU(s) so waiting orders proceed. NOT a trip.
        for pid, (pod, sku_ids) in pod_fills.items():
            for sku_id in sku_ids:
                d = pod.skus[sku_id]
                added = d['limit_qty'] - d['current_qty']
                if added <= 0:
                    continue
                d['current_qty'] = d['limit_qty']             # refill stocked-out SKU only
                self.pod_manager.restore_sku_data(sku_id, added)
            pod.mass = sum(x['weight'] * x['current_qty'] for x in pod.skus.values())
            self._v13_net_fetches += 1            # emergency event counter (NOT a trip)
        self._stockout_skus -= skus_to_remove

    def _compute_pending_demand(self) -> dict:
        """
        Hitung total units per SKU yang masih dibutuhkan oleh orders yang sudah
        di-assign ke picking station tapi belum di-deliver.
        Ini adalah "committed demand" — sudah pasti akan mengurangi stok segera.
        """
        pending = {}
        for order in self.order_manager.unfinished_orders:
            if order.station_id is None:
                continue
            for sku, details in order.skus.items():
                remaining = details['total_quantity'] - details['quantity_delivered']
                if remaining > 0:
                    pending[sku] = pending.get(sku, 0) + remaining
        return pending

    def _dispatch_stockout_pods(self):
        """Per scan interval: for each stockout SKU, dispatch at most 1 pod if no pod already dispatched for that SKU."""
        skus_to_remove = set()
        for sku_id in list(self._stockout_skus):
            # Skip if any pod for this SKU is already non-idle (already on the way to replenishment)
            pods_for_sku = self.pod_manager.sku_to_pods.get(sku_id) or []
            already_en_route = any(
                p is not None
                and not self.pod_manager.is_idle(p.pod_id)
                and getattr(p, 'last_trigger', None) == 'stockout'
                for p in pods_for_sku
            )
            if already_en_route:
                skus_to_remove.add(sku_id)
                continue
            for pod in pods_for_sku:
                if pod is None or not self.pod_manager.is_idle(pod.pod_id):
                    continue
                if sku_id not in pod.skus or pod.skus[sku_id]['current_qty'] >= pod.skus[sku_id]['limit_qty']:
                    continue
                station_replenish = self.station_manager.find_available_replenish_station()
                if station_replenish is None:
                    return  # no station capacity — stop entirely
                nearest_robot = None
                current_distance = float('inf')
                for o in self.get_movable_objects():
                    if o.object_type == 'robot' and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                        dist = calculateDistance(o.pos_x, o.pos_y, pod.pos_x, pod.pos_y)
                        if dist < current_distance:
                            nearest_robot = o
                            current_distance = dist
                if nearest_robot is None:
                    break  # no idle robot — try next SKU
                latest_pod_location = get_pod_location(pod.pod_id)
                if latest_pod_location:
                    pod.pos_x, pod.pos_y = int(latest_pod_location[0]), int(latest_pod_location[1])
                station_replenish.add_pod(pod.pod_id)
                new_job = RobotJob(pod.coordinate, station_id=station_replenish.station_id, pod=pod)
                new_job.add_replenishment_task(pod)
                pod.last_trigger = 'stockout'
                # Stockout override fills ONLY the stocked-out SKU on arrival (not
                # the whole pod): a reactive trip should not "freebie top-up" the
                # rest of the pod, which would understate the baseline's true trip
                # count. The triggering SKU is recorded so finish_replenishment_task
                # refills just that SKU. (Proactive ROP/opportunity trips still
                # top-up the whole pod — see finish_replenishment_task.)
                pod.stockout_fill_skus = set(getattr(pod, 'stockout_fill_skus', set()))
                pod.stockout_fill_skus.add(sku_id)
                nearest_robot.assign_job_and_set_move_to_take_pod(new_job)
                self.pod_manager.mark_pod_not_available(pod)
                skus_to_remove.add(sku_id)  # 1 pod dispatched, done for this SKU
                break
        self._stockout_skus -= skus_to_remove

    def _dispatch_proactive_replenishment(self, extra_pod=None, picked_skus=None):
        """Post-pick ORS dispatcher — routes by ORS_VERSION (6=two-level ROP pod-local,
        7=+in-flight reservation; v6 is the default).
        v8/v9 are NOT routed here — they are purely proactive (loop-tick) and
        finish_picking_task returns before reaching this dispatcher."""
        if ORS_VERSION == 7:
            return self._dispatch_proactive_replenishment_v7(extra_pod=extra_pod,
                                                             picked_skus=picked_skus)
        return self._dispatch_proactive_replenishment_v6(extra_pod=extra_pod,
                                                         picked_skus=picked_skus)

    def _pod_critical_coverage(self, pod, critical_set, pending):
        """Critical SKUs this pod can actually restock now: globally critical
        (s ∈ critical_set), this pod's slot also low (slot_stock ≤ rop_per_slot),
        and room to refill (current < limit). Returns {sku: (urgency, room)} where
        urgency = committed pending demand (fallback mean_daily_demand), room = fill."""
        cov = {}
        for s_id in critical_set:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            limit = float(s.get('limit_qty', 0))
            if cur >= limit:
                continue                       # slot full — cannot help this SKU
            if not Pod.is_slot_low(s):
                continue                       # per-slot ROP filter: no slot low
            committed = float(pending.get(s_id, 0))
            urgency = committed if committed > 0 else float(
                self.pod_manager.skus_data.get(s_id, {}).get('mean_daily_demand', 0.0))
            cov[s_id] = (urgency, limit - cur)
        return cov

    def _reserve_pod_fill(self, pod, covered_skus):
        """Reserve the in-flight refill for the covered SKUs (v7-style), so other
        pods' triggers for these SKUs are suppressed while this pod is en route.

        The reserved amount must equal what the trip will ACTUALLY add at the
        replenishment station = fill_rate × limit_qty − current_qty (the trip fills
        up to V12_FILL_RATE × limit, not to full). Reserving the full gap
        (limit − current) while only filling fill_rate×limit over-claims when
        fill_rate < 1: the SKU is suppressed from the critical set as if it will be
        refilled to full, but it lands below ROP, so it goes critical again sooner —
        inflating trips and stockouts. Reserving the real fill keeps the critical-set
        suppression exactly matched to the refill, so fill_rate sweeps are fair.
        (For fill_rate = 1 this is identical to the old limit − current.)

        v15/v16/v19 (whole-pod fill) reserve EVERY SKU in the pod, not just the covered
        (critical) ones, so the reservation exactly matches the whole-pod refill that
        the trip will perform (reserve = fill). For all other versions the reservation
        is over the covered critical SKUs only (matching their critical-only fill)."""
        if ORS_VERSION in (15, 16, 19, 20):
            covered_skus = list(pod.skus.keys())
        pod.reserved_fill = {}
        for s_id in covered_skus:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            target = V12_FILL_RATE * float(s.get('limit_qty', 0))
            fill = target - float(s.get('current_qty', 0))
            if fill > 0:
                self.reserved_global[s_id] += fill
                pod.reserved_fill[s_id] = fill

    def _build_critical_set(self):
        """Globally-critical SKU set: warehouse stock at/under ROP (ROP-global trigger),
        net of in-flight reservations so a SKU an en-route pod already covers drops out."""
        critical = set()
        for s_id, d in self.pod_manager.skus_data.items():
            eff = d.get('current_global_qty', 1) + self.reserved_global.get(s_id, 0.0)
            if eff <= d.get('rop_global', 0):
                critical.add(s_id)
        return critical

    def _v9_pod_score(self, pod, critical_set, pending):
        """v9 opportunity score of a pod: Σ pending(s)·urgency(s) over SKUs that are
        critical globally (s ∈ critical_set) AND low in THIS pod's slot
        (slot_stock ≤ rop_per_slot). urgency = 1 − slot_stock/rop_per_slot ∈ [0,1]
        measures how close the slot is to empty (proportional, so it is fair between
        large-box and small-box SKUs). pending = committed waiting-order units for the
        SKU (relevance to throughput); SKUs with no waiting demand (pending = 0)
        contribute nothing, so a trip is only worthwhile if it serves real orders.
        Returns (score, covered_skus)."""
        score = 0.0
        covered = []
        for s_id in critical_set:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            rps = float(s.get('rop_per_slot', 0))
            if not Pod.is_slot_low(s):
                continue                       # this pod's slot is not low for s
            if cur >= float(s.get('limit_qty', 0)):
                continue                       # slot already full — refill adds nothing
            pend = float(pending.get(s_id, 0))
            if pend <= 0:
                continue                       # no waiting order needs s right now
            urgency = max(0.0, 1.0 - Pod.slot_stock(s) / rps) if rps > 0 else 0.0
            score += pend * urgency
            covered.append(s_id)
        return score, covered

    def _v10_pod_score(self, pod, critical_set, pending):
        """v10 opportunity score (scan-based, periodic global review).

            Score(p) = Σ  g_i · (1 − cur_i/lim_i) · pending_i
                     i ∈ critical_set ∩ p,  cur_i < lim_i,  pending_i > 0

        where, for SKU i:
          g_i = (rop_global_i − current_global_i) / rop_global_i, clipped [0,1]
              — warehouse urgency: how deep below ROP the SKU is (per-SKU ranking,
                SKUs closer to stockout weigh more), straight from current_global;
          (1 − cur_i/lim_i) — slot fill-rate deficit in THIS pod, normalized so it
              is fair across item box sizes (no unit/box-size bias);
          pending_i — committed units awaited by active orders (demand-awareness):
              SKUs with no waiting order contribute 0, so trips are spent only on
              SKUs orders actually need now — this is what keeps trips low
              (a pure stock-based score over-replenishes SKUs not yet demanded).
        Σ means a pod serving more (urgent, depleted, demanded) critical SKUs ranks
        higher — joint replenishment fills the whole pod, so more useful SKUs per
        trip is worth more.

        Trigger is ROP-global only (membership in critical_set); slot emptiness and
        demand enter as WEIGHTS, not gates (no rop_per_slot / is_slot_low — tested
        and it did not reduce trips). This is v9's demand-awareness PLUS the global
        urgency weight g_i.
        Returns (score, covered_skus)."""
        skus_data = self.pod_manager.skus_data
        score = 0.0
        covered = []
        for s_id in critical_set:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            lim = float(s.get('limit_qty', 0))
            if lim <= 0 or cur >= lim:
                continue                       # slot full — refill adds nothing
            pend = float(pending.get(s_id, 0))
            if pend <= 0:
                continue                       # no waiting order needs this SKU now
            d = skus_data.get(s_id, {})
            rg = float(d.get('rop_global', 0))
            cg = float(d.get('current_global_qty', 0))
            g = max(0.0, min(1.0, (rg - cg) / rg)) if rg > 0 else 0.0
            score += g * (1.0 - cur / lim) * pend
            covered.append(s_id)
        return score, covered

    def _v11_pod_score(self, pod, critical_set):
        """v11 opportunity score WITHOUT pending (ablation): Σ g_i·(1−cur/lim) over
        critical SKUs in the pod with cur < lim. Pure stock-based (no demand-awareness)
        — replenishes any critical, depleted SKU regardless of waiting orders.
        Returns (score, covered_skus)."""
        skus_data = self.pod_manager.skus_data
        score = 0.0
        covered = []
        for s_id in critical_set:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            lim = float(s.get('limit_qty', 0))
            if lim <= 0 or cur >= lim:
                continue                       # slot full — refill adds nothing
            d = skus_data.get(s_id, {})
            rg = float(d.get('rop_global', 0))
            cg = float(d.get('current_global_qty', 0))
            # Urgency g_i = how deep the warehouse stock is BELOW the reorder point:
            # (rop − current_global) / rop, clipped [0,1]. All scored SKUs are already ≤ ROP
            # (critical-set gate), so g_i = how far past the danger line (0 at ROP, 1 at zero
            # stock). ROP is per-SKU (demand×lead-time) → fair across SKU sizes, unlike cur/max.
            g = max(0.0, min(1.0, (rg - cg) / rg)) if rg > 0 else 0.0
            score += g * (1.0 - cur / lim)
            covered.append(s_id)
        return score, covered

    def _v17_pod_score(self, pod, critical_set):
        """v17 = v12 with CONVEX urgency g_i². IDENTICAL to _v11_pod_score (same critical
        gate, same emptiness factor 1−cur/lim, same Σ over the pod) EXCEPT g_i → g_i². The
        square sharpens the spread: a near-stockout SKU (g≈1) keeps its full weight while a
        merely-past-ROP SKU (g≈0.3) is squashed to ~0.09. This lets a pod carrying ONE very
        near-stockout SKU beat a pod carrying MANY mildly-critical SKUs (the SUM-bias the
        linear g_i suffers) — so an at-the-station picked pod holding a dying SKU is no longer
        out-summed by an idle pod full of mild ones. v12 (g_i linear) stays untouched."""
        skus_data = self.pod_manager.skus_data
        score = 0.0
        covered = []
        for s_id in critical_set:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            lim = float(s.get('limit_qty', 0))
            if lim <= 0 or cur >= lim:
                continue
            d = skus_data.get(s_id, {})
            rg = float(d.get('rop_global', 0))
            cg = float(d.get('current_global_qty', 0))
            g = max(0.0, min(1.0, (rg - cg) / rg)) if rg > 0 else 0.0
            score += (g * g) * (1.0 - cur / lim)
            covered.append(s_id)
        return score, covered

    def _v19_pod_score(self, pod, critical_set):
        """v19 = opportunity-cost à la Hsiao Eq 3 (Σ priority × rep-capacity, NO
        divisor), summed over ALL SKUs in the pod (not just the critical set, so the
        whole pod is considered — consistent with whole-pod fill). For each SKU:
            priority_i = max(0, rop_i − current_global_i)   (UNITS, ROP-based)
            capacity_i = (limit_i − current_i)              (slot empty space, UNITS)
            contribution = priority_i × capacity_i
        priority is capped at 0 (healthy SKUs with global stock ≥ ROP contribute 0 →
        neutral, NOT negative), so a pod's many healthy SKUs cannot drown its few
        critical ones (the failure mode of Hsiao's raw x_F−SKU at a 20-slot pod).
        EXTENSIVE — no /n_j: the score is the TOTAL opportunity cost of replenishing
        the whole pod in one trip (unlike Tracy's size-normalised SP_j = ΣK_i/n_j).
        `critical_set` is accepted for signature compatibility but not used (the loop
        is over the whole pod). Returns (score, covered) where covered = the SKUs that
        contributed (priority > 0), i.e. the critical ones."""
        skus_data = self.pod_manager.skus_data
        score = 0.0
        covered = []
        for s_id, s in pod.skus.items():
            cur = float(s.get('current_qty', 0))
            lim = float(s.get('limit_qty', 0))
            # Capacity = empty space to the FULL slot limit (lim − cur), independent of
            # the fill rate φ. φ is an ENERGY lever (pod mass = how much we fill) and must
            # NOT enter the score: the score only RANKS pods (relative), and lim−cur ranks
            # correctly (most-critical + most-empty pod wins). Putting φ here made the fill
            # rate also change WHICH pod is chosen (slots above φ×lim get skipped → ranking
            # shifts per φ) → non-monotone/zigzag throughput across fill rates that can't be
            # explained. Keep pod-selection (which pod) and fill rate (how much) separate.
            cap = lim - cur
            if cap <= 0:
                continue                       # slot full — refill adds nothing
            d = skus_data.get(s_id, {})
            rg = float(d.get('rop_global', 0))
            cg = float(d.get('current_global_qty', 0))
            priority = max(0.0, rg - cg)       # UNITS, capped at 0 (healthy → 0)
            if priority <= 0:
                continue                       # healthy SKU: contributes nothing
            score += priority * cap
            covered.append(s_id)
        return score, covered

    def _v20_pod_score(self, pod, critical_set):
        """v20 = MIN variant of v19 (deliverable-shortage interpretation). Same
        structure as v19 (Σ over ALL SKUs, ROP-based global shortage, healthy SKUs
        contribute 0), but each SKU's contribution is the shortage that can ACTUALLY
        be closed by this pod's empty space in ONE trip:
            contribution_i = min( max(0, rop_i − current_global_i),  (limit_i − cur_i) )
        i.e. min(global shortage, local empty space). Unlike v19's product
        (priority × capacity), the MIN never over-rewards: a SKU short by 100 with
        only 5 empty slots contributes 5, not 500. The score is then the total units
        of critical shortage this pod can immediately cover — a physically meaningful
        quantity (interpretable in raw units, not units²). Trade-off vs v19: MIN
        truncates urgency at the slot gap, so a very-critical SKU (large shortage) is
        treated the same as a mildly-critical one once both exceed the gap — it does
        NOT preserve the full priority ordering the way v19's product does. Added as a
        SEPARATE comparator (v19 and all other versions untouched); not Hsiao-derived,
        so it is positioned as an intuitive deliverable-coverage heuristic.
        Returns (score, covered) where covered = SKUs with positive contribution."""
        skus_data = self.pod_manager.skus_data
        score = 0.0
        covered = []
        for s_id, s in pod.skus.items():
            cur = float(s.get('current_qty', 0))
            lim = float(s.get('limit_qty', 0))
            cap = lim - cur
            if cap <= 0:
                continue                       # slot full — refill adds nothing
            d = skus_data.get(s_id, {})
            rg = float(d.get('rop_global', 0))
            cg = float(d.get('current_global_qty', 0))
            shortage = max(0.0, rg - cg)       # UNITS, capped at 0 (healthy → 0)
            if shortage <= 0:
                continue                       # healthy SKU: contributes nothing
            score += min(shortage, cap)        # deliverable shortage this trip
            covered.append(s_id)
        return score, covered

    def _pstockout_pod_score(self, pod, critical_set):
        """v16 STOCKOUT-PROBABILITY score: Σ P_i · (1 − cur/lim) over the pod's critical
        SKUs with cur < lim. Identical structure to _v11_pod_score (same emptiness factor
        1 − cur/lim, same Σ over critical SKUs that have room) EXCEPT the urgency term is the
        true stockout probability instead of the linear g_i:

            P_i = P(demand during lead time L > current warehouse stock)
                = 1 − Φ( (s − μ_h·L) / (σ_h·√L) )

        μ_h, σ_h are the per-hour demand mean/std that ALSO compute rop_global
        (rop = μ_h·L + Φ⁻¹(SL)·σ_h·√L), so the score and the rop gate share one model. P_i
        is convex in s near depletion — a SKU well below μ_h·L gets ~1.0, one just past ROP
        gets ~(1 − SL) — so near-stockout SKUs stand out far more than under the linear g_i.
        Unlike Tracy (SP_j = ΣK_i/n_j, an average of inventory-level RANKS), this is a real
        probability summed (not averaged) over the pod, so multi-SKU consolidation is
        rewarded. Returns (score, covered_skus)."""
        from scipy.stats import norm
        skus_data = self.pod_manager.skus_data
        L = LEAD_TIME_HOURS
        sqrtL = math.sqrt(L)
        score = 0.0
        covered = []
        for s_id in critical_set:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            lim = float(s.get('limit_qty', 0))
            if lim <= 0 or cur >= lim:
                continue                       # slot full — refill adds nothing
            d = skus_data.get(s_id, {})
            cg = float(d.get('current_global_qty', 0))
            mu_L = float(d.get('mean_hourly_demand', 0.0)) * L
            sig_L = float(d.get('std_hourly_demand', 0.0)) * sqrtL
            if sig_L <= 0:
                # No demand variability → deterministic: stockout iff stock ≤ expected demand.
                p = 1.0 if cg <= mu_L else 0.0
            else:
                p = 1.0 - float(norm.cdf((cg - mu_L) / sig_L))
            score += p * (1.0 - cur / lim)
            covered.append(s_id)
        return score, covered

    def _v13_pod_score(self, pod, critical_set):
        """v13 opportunity score — WEIGHTED SUM of two components (analogous to the
        J = w_o·√OVR + w_i·√IVR cost function of Priore et al., 2019, which balances
        two objectives via weights). Both components are scaled 0–1.

            Score_p = W_G · g_pod + W_E · emptiness_pod

          g_pod (global urgency / anti-stockout) = MAX over the pod's CRITICAL SKUs of
                g_i = max(0, (rop_global_i − current_global_i) / rop_global_i)
              — how close the most-critical SKU is to a WAREHOUSE-level stockout
              (relative to its reorder point). Pulls the pod UP when a SKU is near a
              global stockout. Range 0–1. SERVICE component (OVR-analog): evaluated over
              critical SKUs because those are the ones at stockout risk.

          emptiness_pod (whole-pod slot depletion) = 1 − (Σ current_qty / Σ limit_qty)
              over ALL SKUs in the pod — how empty THIS pod is overall, i.e. how much
              room a replenishment trip would actually refill. Evaluated over the WHOLE
              pod (not just critical SKUs) because a replenishment trip refills the
              ENTIRE pod (replenish_all_skus), so trip efficiency depends on the pod's
              TOTAL deficit, not just its critical slots. Range 0–1. TRIP-EFFICIENCY
              component (IVR-analog).

        The two look at DIFFERENT levels — g at the warehouse, emptiness at this pod's
        total fill — so they are complementary, not duplicates:
          W_G high → prefer the pod whose SKU is most globally critical (service).
          W_E high → prefer the pod that is most empty overall (most room to refill →
                     fuller, more efficient trips → fewer total trips).
        Varying (W_G, W_E) shifts the winning pod, tracing a Pareto frontier.

        The pod is scored only if it covers ≥1 critical SKU with room to refill.
        Returns (score, covered_critical_skus)."""
        skus_data = self.pod_manager.skus_data
        covered = []
        # --- g component: COMBINED stockout risk over CRITICAL SKUs (service) ---
        # g = 1 − ∏(1 − g_i) over the pod's critical SKUs, where g_i is each SKU's urgency
        # (rop−cg)/rop clipped to [0,1] (per-SKU formula UNCHANGED). Unlike a MAX — which only
        # sees the single most-critical SKU and is blind to how many critical SKUs a pod
        # carries — the product form is coverage-aware: a pod carrying MORE (or more urgent)
        # critical SKUs scores higher, so replenishing it relieves more impending stockouts in
        # one trip. It stays in [0,1] (same scale as emptiness, so the W_G/W_E weights remain
        # interpretable), while still rising with the number of critical SKUs. Reads as the
        # pod's combined stockout risk.
        prod_safe = 1.0                        # ∏(1 − g_i)
        for s_id in critical_set:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            lim = float(s.get('limit_qty', 0))
            if lim <= 0 or cur >= lim:
                continue                       # slot full — refill adds nothing
            d = skus_data.get(s_id, {})
            rg = float(d.get('rop_global', 0))
            cg = float(d.get('current_global_qty', 0))
            # g_i = how deep the warehouse stock is BELOW the reorder point, relative to ROP:
            # (rop − current_global) / rop, clipped [0,1]. All scored SKUs are already ≤ ROP
            # (the critical-set gate), so this measures HOW FAR past the danger line each SKU
            # is — g_i=0 at ROP, g_i=1 at zero stock. ROP is per-SKU (demand×lead-time), so it
            # is a fair urgency yardstick across SKUs of different size, unlike cur/max which
            # over-weights large-capacity SKUs.
            g_i = max(0.0, min(1.0, (rg - cg) / rg)) if rg > 0 else 0.0
            prod_safe *= (1.0 - g_i)
            covered.append(s_id)
        if not covered:
            return 0.0, covered
        g_pod = 1.0 - prod_safe                # combined stockout risk, [0,1], rises with coverage
        # --- emptiness component: POD INVENTORY LEVEL over the WHOLE POD ---
        # emptiness_pod = 1 − (Σ current_qty / Σ limit_qty) over ALL SKUs in the pod
        # (critical AND non-critical) = how empty the WHOLE pod is. The trip refills the
        # WHOLE pod (V12_FILL_WHOLE_POD), so trip efficiency depends on the pod's TOTAL
        # deficit, not just critical slots. Measured over all SKUs, this is INDEPENDENT of
        # g (which only looks at critical SKUs at the warehouse level) — so the W_G/W_E
        # weights actually move the winning pod (the critical-only emptiness was ~0.95
        # correlated with g → flat Pareto). Σcur/Σlim is quantity-weighted = the pod's
        # real fill level.
        tot_cur = 0.0
        tot_lim = 0.0
        for s in pod.skus.values():
            lim = float(s.get('limit_qty', 0))
            if lim <= 0:
                continue
            tot_cur += float(s.get('current_qty', 0))
            tot_lim += lim
        emptiness_pod = (1.0 - tot_cur / tot_lim) if tot_lim > 0 else 0.0
        score = V13_W_G * g_pod + V13_W_E * emptiness_pod
        # Stash the two components on the pod so the dispatcher can log them for the
        # WINNING pod → lets us measure the real g–emptiness correlation (how independent
        # service vs efficiency objectives are → whether the Pareto front will have spread
        # or collapse to a line). Diagnostic only.
        pod._last_g_pod = g_pod
        pod._last_emptiness_pod = emptiness_pod
        return score, covered

    def _pod_opportunity_v8(self, pod, critical_set, pending):
        """v8 opportunity score:
            OpportunityScore_p = CriticalDemandCoverage_p × TotalRefillGap_p
        where
            CriticalDemandCoverage_p = Σ_{i critical in p} min(pending_i, refill_gap_i,p)
                — committed order demand this pod can actually satisfy for its critical
                  SKUs (capped by what its slots can hold, via min);
            TotalRefillGap_p = Σ_{all SKUs j in p} (slot_capacity_j − current_qty_j)
                — total empty space a single trip would refill (emptiest-first, whole pod).
        Returns (score, covered_critical_skus). A SKU is 'critical' iff it passes the
        two-level trigger (global ≤ rop_global AND per-slot ≤ rop_per_slot) and is awaited
        by an order (pending > 0)."""
        coverage = 0.0
        covered = []
        for s_id in critical_set:
            s = pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            limit = float(s.get('limit_qty', 0))
            if not Pod.is_slot_low(s):
                continue                       # per-slot ROP: this slot not low
            gap_i = limit - cur
            if gap_i <= 0:
                continue                       # slot full
            pend = float(pending.get(s_id, 0))
            if pend <= 0:
                continue                       # no waiting order needs it
            coverage += min(pend, gap_i)       # demand this pod can actually fill
            covered.append(s_id)
        if not covered:
            return 0.0, []
        total_gap = 0.0                        # whole-pod empty space (all SKUs)
        for s in pod.skus.values():
            total_gap += max(0.0, float(s.get('limit_qty', 0)) - float(s.get('current_qty', 0)))
        return coverage * total_gap, covered

    def _v9_select_replenishment_pod(self, exclude_pods=None, use_score=True):
        """EVENT-DRIVEN ARBITRATION: pick the single best IDLE pod to replenish now,
        among pods carrying globally+per-pod critical SKUs needed by waiting orders.
        Returns (pod, covered_skus) or (None, None) if nothing is worthwhile.
        reserved_global already removes SKUs that an en-route pod will cover, so this
        naturally stops diverting robots once each critical SKU has one pod on the way.

        Both require the SAME trigger: the pod must carry ≥1 SKU that is critical at the
        global (≤ rop_global) AND per-slot (≤ rop_per_slot) level. They differ only in how
        eligible pods are RANKED:
          use_score=True  (v9): Σ pending·urgency over the pod's critical SKUs.
          use_score=False (v8): committed-demand-weighted emptiest-first — rank the pod
            by (Σ pending over its critical SKUs) × (pod space ratio). Both factors are
            per-pod, so they combine consistently: the first favours pods whose critical
            SKUs are actually awaited by orders (relevance to throughput), the second
            favours pods that a single trip can refill the most (emptiest-first
            efficiency, Bolu; Merschformann). No urgency/ROP term in the score, so
            rop_per_slot is used only in the trigger, not double-counted.
        Comparing v8 vs v9 isolates whether the per-SKU urgency score helps."""
        pending = self._compute_pending_demand()
        critical = self._build_critical_set()
        if not critical:
            return None, None
        exclude = exclude_pods or set()
        best_pod, best_rank, best_cov = None, 0.0, None
        for pod in self.pod_manager.pods:
            if pod.pod_id in exclude:
                continue
            if not self.pod_manager.is_idle(pod.pod_id):
                continue
            score, cov = self._v9_pod_score(pod, critical, pending)
            if not cov:
                continue                       # pod carries no eligible critical SKU → not triggered
            if use_score:
                rank = score                                   # v9: Σ pending·urgency
            else:
                pending_pod = self._v9_pending_pod(pod, critical, pending)
                rank = pending_pod * self._pod_space_ratio(pod)  # v8: pending × space ratio
            if rank > best_rank:
                best_pod, best_rank, best_cov = pod, rank, cov
        if best_pod is None:
            return None, None
        return best_pod, best_cov

    def _v9_pending_pod(self, pod, critical_set, pending):
        """Total committed pending demand over the pod's critical SKUs (two-level:
        globally critical AND this pod's slot ≤ rop_per_slot). Used by v8's ranking."""
        total = 0.0
        for s_id, s in pod.skus.items():
            if s_id not in critical_set:
                continue                       # lapis 1: globally critical
            if not Pod.is_slot_low(s):
                continue                       # lapis 2: this pod's slot is low
            total += float(pending.get(s_id, 0))
        return total

    def _pod_space_ratio(self, pod):
        """Emptiest-first metric: total empty space over total slot capacity across ALL
        SKUs in the pod (a replenishment trip refills the whole pod, so the benefit is
        the pod's overall fillable fraction). Returns a value in [0, 1]."""
        empty = 0.0
        cap = 0.0
        for s in pod.skus.values():
            limit = float(s.get('limit_qty', 0))
            cur = float(s.get('current_qty', 0))
            cap += limit
            empty += max(0.0, limit - cur)
        return (empty / cap) if cap > 0 else 0.0

    def _pods_in_picking(self):
        """Pods currently carried by a robot on an UNFINISHED PICKING job. Selecting
        one of these costs no extra trip: it is flagged and diverted to replenishment
        right after its pick completes (finish_picking_task)."""
        pods = {}
        for o in self.get_movable_objects():
            if o.object_type != 'robot':
                continue
            job = getattr(o, 'job', None)
            if job is None or job.is_finished or job.pod is None:
                continue
            station = self.station_manager.get_station_by_id(job.station_id)
            if station is not None and station.is_picker_station():
                pods[job.pod.pod_id] = job.pod
        return pods

    def _directed_opportunity_scan(self):
        """
        ORS v8 — DIRECTED OPPORTUNITY REPLENISHMENT (proactive, greedy set-cover).
        Run on the order-processing cadence (loop-tick). Scans the WHOLE warehouse
        for globally-critical SKUs and greedily picks the pods covering the most
        critical demand per trip:

            critical set  S = { s : eff_global(s) ≤ rop_global(s) }   (ROP-global trigger)
            score(pod)    = Σ_{s∈S covered by pod}  urgency(s) × refill_room(s)
            urgency(s)    = pending_demand(s)  (committed orders; fallback mean_daily_demand)

        CANDIDATES include BOTH idle pods AND pods currently picking:
          - idle pod selected      → robot fetches it now (one extra retrieval trip).
          - picking pod selected    → flagged need_replenishment=True; it diverts to
                                       replenishment right after its pick completes
                                       (no extra trip — already in hand).
        Selection is by score only — a picking pod wins iff it genuinely covers more
        critical demand, in which case the trip is also free. Covered SKUs are
        reserved (reserved_global) so the next greedy step / next scan does not
        dispatch a second pod for them. Repeats until S is covered OR replenishment-
        station capacity is exhausted (idle-pod dispatch also needs an idle robot).
        """
        pending = self._compute_pending_demand()
        remaining = self._build_critical_set()
        if not remaining:
            return

        picking_pods = self._pods_in_picking()  # {pod_id: pod} carried on picking jobs

        while remaining:
            station_replenish = self.station_manager.find_available_replenish_station()
            if station_replenish is None:
                break  # respect replenishment-station capacity

            # Score every candidate pod: idle pods + pods currently picking.
            # A picking pod already flagged for replenishment is skipped (counted once).
            best_pod, best_score, best_cov, best_is_picking = None, 0.0, None, False
            seen = set()
            for pod in self.pod_manager.pods:
                pid = pod.pod_id
                is_picking = pid in picking_pods
                if not is_picking and not self.pod_manager.is_idle(pid):
                    continue                      # busy but not picking (e.g. en route) — skip
                if is_picking and pod.need_replenishment:
                    continue                      # already flagged this scan
                if pid in seen:
                    continue
                seen.add(pid)
                cov = self._pod_critical_coverage(pod, remaining, pending)
                if not cov:
                    continue
                score = sum(u * r for (u, r) in cov.values())
                if score > best_score:
                    best_pod, best_score, best_cov, best_is_picking = pod, score, cov, is_picking
            if best_pod is None:
                break  # no candidate covers any remaining critical SKU

            self._reserve_pod_fill(best_pod, best_cov.keys())
            best_pod.last_opp_score = best_score
            remaining -= set(best_cov.keys())

            if best_is_picking:
                # FREE: divert to replenishment after its pick completes.
                best_pod.need_replenishment = True
                best_pod.last_trigger = 'directed_after_pick'
                with open('score_log.csv', 'a') as _f:
                    _f.write(f"{int(self._tick)},{best_pod.pod_id},{best_score:.6f},"
                             f"0.000000,{best_score:.6f},0.000000\n")
                continue  # no station/robot consumed now; replenish station reserved at finish

            # IDLE: fetch now — needs an idle robot.
            nearest_robot, current_distance = None, float('inf')
            for o in self.get_movable_objects():
                if o.object_type == 'robot' and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                    dist = calculateDistance(o.pos_x, o.pos_y, best_pod.pos_x, best_pod.pos_y)
                    if dist < current_distance:
                        nearest_robot, current_distance = o, dist
            if nearest_robot is None:
                # release the reservation we just made (no robot to carry it)
                for s_id, fill in (best_pod.reserved_fill or {}).items():
                    self.reserved_global[s_id] = max(0.0, self.reserved_global.get(s_id, 0.0) - fill)
                best_pod.reserved_fill = {}
                break

            latest = get_pod_location(best_pod.pod_id)
            if latest:
                best_pod.pos_x, best_pod.pos_y = int(latest[0]), int(latest[1])
            station_replenish.add_pod(best_pod.pod_id)
            new_job = RobotJob(best_pod.coordinate, station_id=station_replenish.station_id, pod=best_pod)
            new_job.add_replenishment_task(best_pod)
            best_pod.last_trigger = 'directed_opportunity'
            nearest_robot.assign_job_and_set_move_to_take_pod(new_job)
            self.pod_manager.mark_pod_not_available(best_pod)
            with open('score_log.csv', 'a') as _f:
                _f.write(f"{int(self._tick)},{best_pod.pod_id},{best_score:.6f},"
                         f"0.000000,{best_score:.6f},0.000000\n")

    def _dispatch_replenishment_event(self, use_score=True, max_dispatch=None):
        """EVENT-DRIVEN replenishment (WES robot life-cycle, Bolu & Korçak 2021): when
        robots are free, send each to replenish the best pod carrying two-level-critical
        SKUs (global ≤ rop_global AND per-slot ≤ rop_per_slot) needed by waiting orders.
        Ranking: v9 = Σ pending·urgency; v8/v7 = CriticalDemandCoverage × TotalRefillGap.
        max_dispatch caps how many pods are sent THIS tick:
          None (v8/v9) = greedy, dispatch all worthwhile pods (bounded by free robots /
                         station capacity);
          1    (v7)     = at most one pod per tick, so replenishment never takes more than
                         one robot from picking per step (more conservative).
        Reservation prevents two pods per SKU; stops when no free robot / no free
        station / no worthwhile pod, so picking isn't starved."""
        pending = self._compute_pending_demand()
        critical = self._build_critical_set()
        if not critical:
            return
        dispatched = 0
        while True:
            if max_dispatch is not None and dispatched >= max_dispatch:
                break  # cap reached (v7: 1 pod/tick)
            station_replenish = self.station_manager.find_available_replenish_station()
            if station_replenish is None:
                break  # respect replenishment-station capacity

            # best idle pod carrying still-critical SKUs (re-evaluated each iteration)
            #   v9 = Σ pending·urgency (_v9_pod_score)
            #   v8 = CriticalDemandCoverage × TotalRefillGap (_pod_opportunity_v8)
            best_pod, best_rank, best_cov = None, 0.0, None
            for pod in self.pod_manager.pods:
                if not self.pod_manager.is_idle(pod.pod_id):
                    continue
                if ORS_VERSION == 10:
                    rank, cov = self._v10_pod_score(pod, critical, pending)
                elif use_score:
                    rank, cov = self._v9_pod_score(pod, critical, pending)
                else:
                    rank, cov = self._pod_opportunity_v8(pod, critical, pending)
                if not cov:
                    continue
                if rank > best_rank:
                    best_pod, best_rank, best_cov = pod, rank, cov
            if best_pod is None:
                break  # nothing worthwhile to replenish

            # nearest idle robot to carry it
            nearest_robot, current_distance = None, float('inf')
            for o in self.get_movable_objects():
                if o.object_type == 'robot' and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                    dist = calculateDistance(o.pos_x, o.pos_y, best_pod.pos_x, best_pod.pos_y)
                    if dist < current_distance:
                        nearest_robot, current_distance = o, dist
            if nearest_robot is None:
                break  # no free robot — picking keeps the rest; retried next tick

            self._reserve_pod_fill(best_pod, best_cov)  # no second pod for these SKUs
            critical -= set(best_cov)                   # drop covered SKUs for the rest of this pass
            latest = get_pod_location(best_pod.pod_id)
            if latest:
                best_pod.pos_x, best_pod.pos_y = int(latest[0]), int(latest[1])
            station_replenish.add_pod(best_pod.pod_id)
            new_job = RobotJob(best_pod.coordinate, station_id=station_replenish.station_id, pod=best_pod)
            new_job.add_replenishment_task(best_pod)
            best_pod.last_trigger = ('event_score' if use_score
                                     else ('event_cap1' if max_dispatch == 1 else 'event_cov'))
            best_pod.last_opp_score = sum(float(pending.get(s, 0)) for s in best_cov)
            nearest_robot.assign_job_and_set_move_to_take_pod(new_job)
            self.pod_manager.mark_pod_not_available(best_pod)
            dispatched += 1
            with open('score_log.csv', 'a') as _f:
                _f.write(f"{int(self._tick)},{best_pod.pod_id},{best_pod.last_opp_score:.6f},"
                         f"0.000000,{best_pod.last_opp_score:.6f},0.000000\n")

    def _dispatch_piggyback_ranked_v11(self, picked_pod=None, picking_robot=None, picked_skus=None):
        """ORS v11 — EVENT-DRIVEN (at finish-picking only), one COMBINED ranking, max 1 pod.

        TRIGGER: evaluation runs only if a JUST-PICKED SKU has crossed its global reorder
        point (current_global ≤ rop_global) — picking is what lowered the warehouse stock,
        so this is the reorder-point trigger fired by the demand event that caused it.
        Pickings that leave all their SKUs above ROP trigger nothing.

        SCOPE: once triggered, the FULL critical set is considered (all globally critical
        SKUs, not just the picked ones), so an urgent SKU in a pod that did not pass
        through picking is not missed. Score the just-picked pod TOGETHER with all idle
        (non-queued) pods carrying critical SKUs (Σ g_i·(1−cur/lim), NO pending), then
        dispatch AT MOST ONE pod — the single top scorer:

          - if the just-picked pod is the top scorer → PIGGYBACK (return True so the main
            loop sends it straight to replenishment for free; no fetch trip);
          - if an idle pod is the top scorer → FETCH it once with the nearest idle robot;
          - if nothing scores > 0 → pod returns to storage.

        Remaining critical SKUs wait for the next picking event (~every 6 ticks). Returns
        True iff the just-picked pod is piggybacked. Pure score (no free bonus).
        """
        if picked_pod is None:
            return False
        critical = self._build_critical_set()
        if not critical:
            return False
        # TRIGGER: only proceed if a just-picked SKU is itself globally critical.
        picked_critical = (set(picked_skus) & critical) if picked_skus else set()
        if not picked_critical:
            return False

        queued_pod_ids = {j.pod.pod_id for j in self.job_queue if j.pod is not None}

        # ONE combined ranking, dispatch AT MOST ONE pod (no greedy loop). Remaining
        # critical SKUs wait for the next picking event.
        candidates = [picked_pod] + [
            p for p in self.pod_manager.pods
            if p.pod_id != picked_pod.pod_id
            and self.pod_manager.is_idle(p.pod_id)
            and p.pod_id not in queued_pod_ids
        ]
        best_pod, best_rank, best_cov = None, 0.0, None
        for pod in candidates:
            rank, cov = self._v11_pod_score(pod, critical)  # no pending
            if not cov:
                continue
            if rank > best_rank:
                best_pod, best_rank, best_cov = pod, rank, cov
        if best_pod is None or best_rank <= 0.0:
            return False                         # nothing worthwhile → pod to storage

        # Winner is the just-picked pod → PIGGYBACK (main loop sends it straight on).
        if best_pod.pod_id == picked_pod.pod_id:
            self._reserve_pod_fill(picked_pod, best_cov)
            picked_pod.last_trigger = 'v11_piggyback'
            picked_pod.last_opp_score = best_rank
            with open('score_log.csv', 'a') as _f:
                _f.write(f"{int(self._tick)},{picked_pod.pod_id},{best_rank:.6f},"
                         f"0.000000,{best_rank:.6f},0.000000\n")
            return True

        # Winner is an idle pod → FETCH it once (needs a station + a free robot).
        station_replenish = self.station_manager.find_available_replenish_station()
        if station_replenish is None:
            return False
        nearest_robot, current_distance = None, float('inf')
        for o in self.get_movable_objects():
            if o.object_type == 'robot' and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                dist = calculateDistance(o.pos_x, o.pos_y, best_pod.pos_x, best_pod.pos_y)
                if dist < current_distance:
                    nearest_robot, current_distance = o, dist
        if nearest_robot is None:
            return False
        self._reserve_pod_fill(best_pod, best_cov)
        latest = get_pod_location(best_pod.pod_id)
        if latest:
            best_pod.pos_x, best_pod.pos_y = int(latest[0]), int(latest[1])
        station_replenish.add_pod(best_pod.pod_id)
        new_job = RobotJob(best_pod.coordinate, station_id=station_replenish.station_id, pod=best_pod)
        new_job.add_replenishment_task(best_pod)
        best_pod.last_trigger = 'v11_fetch'
        best_pod.last_opp_score = best_rank
        nearest_robot.assign_job_and_set_move_to_take_pod(new_job)
        self.pod_manager.mark_pod_not_available(best_pod)
        with open('score_log.csv', 'a') as _f:
            _f.write(f"{int(self._tick)},{best_pod.pod_id},{best_rank:.6f},"
                     f"0.000000,{best_rank:.6f},0.000000\n")
        return False                             # just-picked pod returns to storage

    def _dispatch_piggyback_ranked_v12(self, picked_pod=None, picking_robot=None, picked_skus=None):
        """ORS v12 — EVENT-DRIVEN, ONE pod per picking event (no per-tick scan, no greedy).

        Like v11 (evaluated ONLY at finish-picking, opportunity score Σ g_i·(1−cur/lim),
        NO pending), but dispatches AT MOST ONE pod per picking event — the single
        highest-scoring candidate. Remaining critical SKUs wait for the NEXT picking event
        (which comes ~every 6 ticks, so the backlog clears quickly). This keeps the
        mechanism simple ("one replenishment decision per picking") and spreads robot load
        (at most one robot taken from picking per event).

        At finish-picking: rank the just-picked pod TOGETHER with all idle (non-queued)
        pods carrying critical SKUs. Take the single top scorer:
          - if it is the just-picked pod → PIGGYBACK (return True; robot continues straight
            to replenishment, no fetch trip);
          - if it is an idle pod → FETCH it once with the nearest idle robot; the
            just-picked pod returns to storage (return False);
          - if nothing scores > 0 → pod returns to storage (return False).
        Pure score (no free bonus for the just-picked pod). Reservation prevents a later
        event from dispatching a second pod for the same SKU.
        Returns True iff the just-picked pod is piggybacked.
        """
        if picked_pod is None:
            return False
        critical = self._build_critical_set()
        if not critical:
            return False

        # GATE (piggyback trigger): the just-picked pod must hold AT LEAST ONE globally
        # critical SKU (current_global <= rop_global) for this picking event to trigger a
        # replenishment decision. We check every SKU the pod carries (picked_pod.skus), not
        # only the SKUs taken in this job — the pod is in the robot's hands now, so if it
        # holds any critical SKU it is worth evaluating. Without the gate, every
        # finish-picking re-evaluated the whole warehouse (dispatch tracked the constant
        # picking rate, not the ROP); with it, evaluation fires only when the picked pod is
        # itself critical, tying the trigger to ROP (higher ROP → pods go critical sooner →
        # more triggers).
        if not any(s in critical for s in picked_pod.skus):
            return False

        # Candidate set: the just-picked pod PLUS all idle, non-queued pods carrying ANY
        # globally critical SKU. The gate above already established this picking event is
        # worth a replenishment decision (the just-picked pod holds a critical SKU); from
        # here "look at other pods" means comparing the opportunity score of EVERY pod that
        # carries a critical SKU — not only pods that share the picked pod's SKU — and
        # dispatching the single highest-scoring one (the professor's brief).
        queued_pod_ids = {j.pod.pod_id for j in self.job_queue if j.pod is not None}
        candidates = [picked_pod] + [
            p for p in self.pod_manager.pods
            if p.pod_id != picked_pod.pod_id
            and self.pod_manager.is_idle(p.pod_id)
            and p.pod_id not in queued_pod_ids
        ]
        # v16 uses the true stockout-probability score; v17 uses convex g_i² (sharper
        # urgency, fixes SUM-bias); v12 (same dispatch path) keeps the linear g_i score, so
        # v12 stays an untouched comparator.
        if ORS_VERSION == 16:
            score_fn = self._pstockout_pod_score
        elif ORS_VERSION == 17:
            score_fn = self._v17_pod_score
        else:
            score_fn = self._v11_pod_score
        best_pod, best_rank, best_cov = None, 0.0, None
        for pod in candidates:
            rank, cov = score_fn(pod, critical)   # no pending
            if not cov:
                continue
            if rank > best_rank:
                best_pod, best_rank, best_cov = pod, rank, cov

        if best_pod is None or best_rank <= 0.0:
            return False                         # nothing worthwhile → pod to storage

        # DIAGNOSTIC (observational only — does NOT change the decision): does the chosen
        # pod miss a near-stockout SKU that a LOSING candidate carried? Tests the hypothesis
        # that the score skips pods holding about-to-stockout SKUs.
        NEAR = 0.2
        _sd = self.pod_manager.skus_data
        def _near_stockout_skus(p):
            out = set()
            for s_id in critical:
                s = p.skus.get(s_id)
                if s is None:
                    continue
                d = _sd.get(s_id, {})
                rg = float(d.get('rop_global', 0)); cg = float(d.get('current_global_qty', 0))
                if rg > 0 and cg <= NEAR * rg:
                    out.add(s_id)
            return out
        _winner_near = _near_stockout_skus(best_pod)
        _missed = set()
        for _p in candidates:
            if _p.pod_id == best_pod.pod_id:
                continue
            _missed |= (_near_stockout_skus(_p) - _winner_near)
        with open('score_diag.csv', 'a') as _f:
            _f.write(f"{int(self._tick)},{best_pod.pod_id},{best_rank:.4f},"
                     f"{len(candidates)},{len(_winner_near)},{len(_missed)},"
                     f"{';'.join(str(s) for s in sorted(_missed))}\n")

        # Winner is the just-picked pod → PIGGYBACK (main loop sends it straight on).
        if best_pod.pod_id == picked_pod.pod_id:
            self._reserve_pod_fill(picked_pod, best_cov)
            picked_pod.last_trigger = f'v{ORS_VERSION}_piggyback'
            picked_pod.last_opp_score = best_rank
            with open('score_log.csv', 'a') as _f:
                _f.write(f"{int(self._tick)},{picked_pod.pod_id},{best_rank:.6f},"
                         f"0.000000,{best_rank:.6f},0.000000\n")
            return True

        # Winner is an idle pod.
        # If fetch is disabled (V13_ALLOW_FETCH False), do NOT dispatch it — leave it to
        # be served the next time it passes through picking. The just-picked pod returns
        # to storage. This makes every v12 replenishment a PIGGYBACK (zero fetch trips),
        # matching the v13 fetch-off behaviour. The stockout-net (instant, untracked as a
        # trip) catches any SKU that truly hits zero before its pod passes picking.
        if not V13_ALLOW_FETCH:
            return False
        # FETCH the winning idle pod once. Need a replenishment-station slot AND a robot;
        # if either is missing, DO NOT queue — drop this fetch (return False), the same as
        # the v13 path. (Matches v13 fetch logic so v12/v13 are apple-to-apple.)
        station_replenish = self.station_manager.find_available_replenish_station()
        if station_replenish is None:
            return False
        nearest_robot, current_distance = None, float('inf')
        for o in self.get_movable_objects():
            if o.object_type == 'robot' and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                dist = calculateDistance(o.pos_x, o.pos_y, best_pod.pos_x, best_pod.pos_y)
                if dist < current_distance:
                    nearest_robot, current_distance = o, dist
        if nearest_robot is None:
            return False
        self._reserve_pod_fill(best_pod, best_cov)
        latest = get_pod_location(best_pod.pod_id)
        if latest:
            best_pod.pos_x, best_pod.pos_y = int(latest[0]), int(latest[1])
        station_replenish.add_pod(best_pod.pod_id)
        new_job = RobotJob(best_pod.coordinate, station_id=station_replenish.station_id, pod=best_pod)
        new_job.add_replenishment_task(best_pod)
        best_pod.last_trigger = f'v{ORS_VERSION}_fetch'
        best_pod.last_opp_score = best_rank
        self.pod_manager.mark_pod_not_available(best_pod)
        with open('score_log.csv', 'a') as _f:
            _f.write(f"{int(self._tick)},{best_pod.pod_id},{best_rank:.6f},"
                     f"0.000000,{best_rank:.6f},0.000000\n")
        nearest_robot.assign_job_and_set_move_to_take_pod(new_job)
        return False                             # just-picked pod returns to storage

    def _dispatch_piggyback_ranked_v14(self, picked_pod=None, picking_robot=None, picked_skus=None):
        """ORS v14 — IDENTICAL to v12 (same Σ g·(1−cur/lim) score via _v11_pod_score,
        same _reserve_pod_fill, same V12_FILL_RATE, same piggyback-or-fetch, same
        stockout-net) EXCEPT the candidate set is restricted to pods that share a
        critical SKU with the just-picked pod.

        v12 candidate set = picked pod + EVERY idle pod carrying ANY critical SKU.
        v14 candidate set = picked pod + idle pods carrying ≥1 of the picked pod's OWN
        critical SKUs (picked_critical). Rationale: the picking event went critical on
        the picked pod's SKUs; the relevant alternatives are pods that can serve those
        same SKUs (so a fetch is always tied to the SKU that just depleted), not pods
        carrying unrelated critical SKUs. Each candidate is still scored over the WHOLE
        critical set (so a pod that shares one picked-critical SKU AND carries more
        critical SKUs still gets full consolidation credit). Reservation + fill-rate are
        unchanged, so v14 works with the fill-rate sweep exactly like v12.
        Returns True iff the just-picked pod is piggybacked."""
        if picked_pod is None:
            return False
        critical = self._build_critical_set()
        if not critical:
            return False
        # GATE: picked pod must hold ≥1 critical SKU (same as v12).
        picked_critical = {s for s in picked_pod.skus if s in critical}
        if not picked_critical:
            return False

        # CANDIDATE SET (v14 difference): picked pod + idle, non-queued pods that carry
        # at least one of the PICKED POD's critical SKUs (not any critical SKU).
        queued_pod_ids = {j.pod.pod_id for j in self.job_queue if j.pod is not None}
        candidates = [picked_pod] + [
            p for p in self.pod_manager.pods
            if p.pod_id != picked_pod.pod_id
            and self.pod_manager.is_idle(p.pod_id)
            and p.pod_id not in queued_pod_ids
            and (picked_critical & set(p.skus))      # shares ≥1 picked-critical SKU
        ]
        best_pod, best_rank, best_cov = None, 0.0, None
        for pod in candidates:
            if ORS_VERSION == 19:
                rank, cov = self._v19_pod_score(pod, critical)   # Hsiao opportunity-cost (priority × cap)
            elif ORS_VERSION == 20:
                rank, cov = self._v20_pod_score(pod, critical)   # MIN variant (deliverable shortage)
            else:
                rank, cov = self._v11_pod_score(pod, critical)   # scored over WHOLE critical set (v14/v15)
            if not cov:
                continue
            if rank > best_rank:
                best_pod, best_rank, best_cov = pod, rank, cov

        if best_pod is None or best_rank <= 0.0:
            return False

        # Winner is the just-picked pod → PIGGYBACK.
        if best_pod.pod_id == picked_pod.pod_id:
            self._reserve_pod_fill(picked_pod, best_cov)
            picked_pod.last_trigger = 'v14_piggyback'
            picked_pod.last_opp_score = best_rank
            with open('score_log.csv', 'a') as _f:
                _f.write(f"{int(self._tick)},{picked_pod.pod_id},{best_rank:.6f},"
                         f"0.000000,{best_rank:.6f},0.000000\n")
            return True

        # Winner is an idle pod → FETCH (same logic as v12).
        if not V13_ALLOW_FETCH:
            return False
        station_replenish = self.station_manager.find_available_replenish_station()
        if station_replenish is None:
            return False
        nearest_robot, current_distance = None, float('inf')
        for o in self.get_movable_objects():
            if o.object_type == 'robot' and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                dist = calculateDistance(o.pos_x, o.pos_y, best_pod.pos_x, best_pod.pos_y)
                if dist < current_distance:
                    nearest_robot, current_distance = o, dist
        if nearest_robot is None:
            return False
        self._reserve_pod_fill(best_pod, best_cov)
        latest = get_pod_location(best_pod.pod_id)
        if latest:
            best_pod.pos_x, best_pod.pos_y = int(latest[0]), int(latest[1])
        station_replenish.add_pod(best_pod.pod_id)
        new_job = RobotJob(best_pod.coordinate, station_id=station_replenish.station_id, pod=best_pod)
        new_job.add_replenishment_task(best_pod)
        best_pod.last_trigger = 'v14_fetch'
        best_pod.last_opp_score = best_rank
        self.pod_manager.mark_pod_not_available(best_pod)
        with open('score_log.csv', 'a') as _f:
            _f.write(f"{int(self._tick)},{best_pod.pod_id},{best_rank:.6f},"
                     f"0.000000,{best_rank:.6f},0.000000\n")
        nearest_robot.assign_job_and_set_move_to_take_pod(new_job)
        return False

    def _dispatch_piggyback_ranked_v13(self, picked_pod=None, picking_robot=None, picked_skus=None):
        """ORS v13 — identical mechanism to v12 (event-driven, ONE pod per picking
        event, piggyback-or-fetch) but ranks candidates with the WEIGHTED-SUM score
        Score = W_G·urgency + W_E·emptiness (see _v13_pod_score) instead of v12's
        multiplicative Σ g·(1−cur/lim). The weights trade off anti-stockout (W_G)
        against trip-efficiency (W_E); vary them for a Pareto frontier.

        At finish-picking: rank the just-picked pod together with all idle, non-queued
        pods carrying critical SKUs; take the single top scorer. If it is the
        just-picked pod → PIGGYBACK (True); if an idle pod → FETCH it once (False);
        if nothing scores > 0 → pod returns to storage (False).
        """
        if picked_pod is None:
            return False
        critical = self._build_critical_set()
        if not critical:
            return False

        # GATE (piggyback trigger): the just-picked pod must hold AT LEAST ONE globally
        # critical SKU (current_global <= rop_global) for this picking event to trigger a
        # replenishment decision — same gate as v12, so v12 and v13 differ ONLY in the
        # scoring formula, not in when evaluation fires. Without it, every finish-picking
        # re-evaluated the whole warehouse (dispatch tracked the picking rate, not the ROP).
        if not any(s in critical for s in picked_pod.skus):
            return False

        queued_pod_ids = {j.pod.pod_id for j in self.job_queue if j.pod is not None}
        candidates = [picked_pod] + [
            p for p in self.pod_manager.pods
            if p.pod_id != picked_pod.pod_id
            and self.pod_manager.is_idle(p.pod_id)
            and p.pod_id not in queued_pod_ids
        ]
        best_pod, best_rank, best_cov = None, 0.0, None
        for pod in candidates:
            rank, cov = self._v13_pod_score(pod, critical)
            if not cov:
                continue
            if rank > best_rank:
                best_pod, best_rank, best_cov = pod, rank, cov

        if best_pod is None or best_rank <= 0.0:
            return False                         # nothing worthwhile → pod to storage

        # DIAGNOSTIC: log the winning pod's two score components (g vs emptiness) so we can
        # measure how independent service vs efficiency are (low correlation → Pareto front
        # has spread; high correlation → it collapses to a line).
        with open('v13_components_log.csv', 'a') as _f:
            _f.write(f"{int(self._tick)},{best_pod.pod_id},"
                     f"{getattr(best_pod,'_last_g_pod',0):.6f},"
                     f"{getattr(best_pod,'_last_emptiness_pod',0):.6f},{best_rank:.6f}\n")

        # Winner is the just-picked pod → PIGGYBACK.
        if best_pod.pod_id == picked_pod.pod_id:
            self._reserve_pod_fill(picked_pod, best_cov)
            picked_pod.last_trigger = 'v13_piggyback'
            picked_pod.last_opp_score = best_rank
            with open('score_log.csv', 'a') as _f:
                _f.write(f"{int(self._tick)},{picked_pod.pod_id},{best_rank:.6f},"
                         f"0.000000,{best_rank:.6f},0.000000\n")
            return True

        # Winner is an idle pod.
        # If fetch is disabled, do NOT dispatch it — leave it to be served the next
        # time it passes through picking. The just-picked pod returns to storage. This
        # makes every replenishment a piggyback (zero fetch trips) across all weights.
        if not V13_ALLOW_FETCH:
            return False
        # Otherwise FETCH it once.
        station_replenish = self.station_manager.find_available_replenish_station()
        if station_replenish is None:
            return False
        nearest_robot, current_distance = None, float('inf')
        for o in self.get_movable_objects():
            if o.object_type == 'robot' and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                dist = calculateDistance(o.pos_x, o.pos_y, best_pod.pos_x, best_pod.pos_y)
                if dist < current_distance:
                    nearest_robot, current_distance = o, dist
        if nearest_robot is None:
            return False
        self._reserve_pod_fill(best_pod, best_cov)
        latest = get_pod_location(best_pod.pod_id)
        if latest:
            best_pod.pos_x, best_pod.pos_y = int(latest[0]), int(latest[1])
        station_replenish.add_pod(best_pod.pod_id)
        new_job = RobotJob(best_pod.coordinate, station_id=station_replenish.station_id, pod=best_pod)
        new_job.add_replenishment_task(best_pod)
        best_pod.last_trigger = 'v13_fetch'
        best_pod.last_opp_score = best_rank
        nearest_robot.assign_job_and_set_move_to_take_pod(new_job)
        self.pod_manager.mark_pod_not_available(best_pod)
        with open('score_log.csv', 'a') as _f:
            _f.write(f"{int(self._tick)},{best_pod.pod_id},{best_rank:.6f},"
                     f"0.000000,{best_rank:.6f},0.000000\n")
        return False                             # just-picked pod returns to storage

    def _dispatch_proactive_replenishment_v6(self, extra_pod=None, picked_skus=None):
        """
        ORS v6 — TWO-LEVEL ROP, pod-local, no candidate search, no ranking, no gate.

          Trigger (scope = just-PICKED SKUs only — the SKUs whose pod stock just
          dropped; other SKUs in the pod are unchanged since the last visit, so
          rechecking them is redundant):
              critical(s)  <=>  current_global(s)  <= rop_global(s)        (warehouse ROP)
                            AND  slot_stock(s)     <= rop_per_slot(s)       (per-slot ROP)
          Both reorder points must be reached: the warehouse stock of SKU s is at its
          global reorder point AND this pod's own slot for s is at its per-slot reorder
          point. The two-level AND is the entire admission test — it replaces the
          theta gate. It is justifiable as classic (s) reorder logic applied at two
          inventory levels (warehouse + pod), not a tuned parameter.

          Action: if any just-picked SKU is critical (and still fillable), replenish
          THAT pod directly (it is already at hand — no robot is sent to a different
          pod, so no extra transport trip). No "look at other pods", no ranking.
          Cheapest possible mechanism — minimises replenishment trips, the right
          direction in this robot-bound regime.

        Returns True if extra_pod is selected for replenishment, False otherwise.
        """
        if extra_pod is None:
            return False

        # total_space over ALL SKUs (refill-to-full worth of this trip, for logging).
        total_space = sum(
            max(0.0, float(s.get('limit_qty', 0)) - float(s.get('current_qty', 0)))
            for s in extra_pod.skus.values()
        )

        # Two-level ROP over the just-PICKED SKUs only.
        scope = set(picked_skus) if picked_skus else set(extra_pod.skus.keys())
        critical = False
        for s_id in scope:
            s = extra_pod.skus.get(s_id)
            if s is None:
                continue
            cur = float(s.get('current_qty', 0))
            limit = float(s.get('limit_qty', 0))
            if cur >= limit:
                continue  # slot full — refilling it does nothing
            if TRIGGER_ON_EMPTY_ONLY:
                # Extreme lower-bound (A2): two-level ROP with BOTH thresholds = 0.
                # Reorder only when the SKU is depleted globally (current_global <= 0)
                # AND every slot is empty in this pod (per-slot stock <= 0). No safety
                # stock at either level. Theoretical worst case.
                d_glob = self.pod_manager.skus_data.get(s_id, {})
                global_empty = d_glob.get('current_global_qty', 1) <= 0
                if global_empty and Pod.slot_stock(s) <= 0:
                    critical = True
                continue
            d = self.pod_manager.skus_data.get(s_id, {})
            global_ok = d.get('current_global_qty', 1) <= d.get('rop_global', 0)
            # Per-slot ROP: at least one slot of this SKU in this pod is at/under
            # rop_per_slot (per-slot stock = current_qty / n_slots_in_pod).
            per_slot_ok = Pod.is_slot_low(s)
            if global_ok and per_slot_ok:
                critical = True

        if not critical:
            return False

        extra_pod.last_opp_score = total_space
        extra_pod.last_trigger = 'two_level_rop'
        with open('score_log.csv', 'a') as _f:
            _f.write(
                f"{int(self._tick)},{extra_pod.pod_id},{total_space:.6f},"
                f"0.000000,{total_space:.6f},0.000000\n"
            )
        return True

    def _dispatch_proactive_replenishment_v7(self, extra_pod=None, picked_skus=None):
        """
        ORS v7 — PIGGYBACK + OPPORTUNITY SCORE.

        The just-picked pod (extra_pod) is already in the robot's hands at the
        station. v7 decides — AT THIS MOMENT (finish-picking) — whether to send it
        straight to replenishment (piggyback, the robot continues; NO fetch trip) or
        let it return to storage. Because the decision is made when the pod is already
        free, nothing is held back — this avoids the "pod tied up" failure of the old
        v7 (which reserved pods while they were still being picked, stalling them).

        Decision rule — piggyback iff the pod carries ≥1 SKU that is BOTH:
            (i)  critical at the warehouse level:  effective_global(s) ≤ rop_global(s)
                 with effective_global(s) = current_global(s) + reserved_global(s)
                 (a SKU an in-flight pod already restocks above ROP is NOT critical →
                  no redundant second trip), AND
            (ii) low in THIS pod's slot:           slot_stock(s) ≤ rop_per_slot(s)
                 with room to refill (current < limit), AND
            (iii) actually awaited by a waiting order: pending(s) > 0.

        The opportunity score (Σ pending·urgency, via _v9_pod_score) is used to RANK
        whether the piggyback is worthwhile: score > 0 means this pod can serve real
        committed demand for a critical SKU. Pods carrying critical SKUs that NO order
        is waiting on (pending = 0) score 0 and are NOT diverted — the same relevance
        gate as v9, so v7 piggybacks exactly the pods v9 would have fetched, but for
        free when they happen to pass through picking. SKUs critical but not passing
        through picking are still served by the event-driven fetch scan in the main
        loop (_dispatch_replenishment_event).

        Returns True if extra_pod is sent to replenishment, False otherwise.
        """
        if extra_pod is None:
            return False

        # Opportunity score over the pod's critical-AND-awaited SKUs. Reuses the v9
        # scorer so v7 (piggyback) and v9 (fetch) apply an identical relevance test;
        # the only difference is v7 gets the pod for free (already at the station).
        critical = self._build_critical_set()
        if not critical:
            return False
        pending = self._compute_pending_demand()
        score, covered = self._v9_pod_score(extra_pod, critical, pending)
        if not covered or score <= 0.0:
            return False                        # nothing critical+awaited → return pod

        # Reserve the in-flight refill for exactly the covered SKUs, so other pods'
        # triggers (and the fetch scan) are suppressed for SKUs this pod will restock
        # — preventing a redundant second trip for the same popular SKU.
        self._reserve_pod_fill(extra_pod, covered)

        total_space = sum(
            max(0.0, float(s.get('limit_qty', 0)) - float(s.get('current_qty', 0)))
            for s in extra_pod.skus.values()
        )
        extra_pod.last_opp_score = score
        extra_pod.last_trigger = 'piggyback_score'
        with open('score_log.csv', 'a') as _f:
            _f.write(
                f"{int(self._tick)},{extra_pod.pod_id},{score:.6f},"
                f"0.000000,{total_space:.6f},0.000000\n"
            )
        return True

    def insert_finished_order_to_csv(self, order: Order):
        header = ["order_id", "order_arrival", "process_start_time", "order_complete_time", "station_id"]
        data = [order.order_id, order.order_arrival, order.process_start_time, order.order_complete_time,
                order.station_id]

        self.write_to_csv("order-finished.csv", header, data)

    def find_new_orders(self):
        file_path = 'assign_order.csv'
        if os.path.exists(file_path):
            assign_order_df = pd.read_csv(file_path)
            # pass
        else:
            orders_df = pd.read_csv('generated_order.csv')
            assign_order_df = orders_df.copy()
            assign_order_df['assigned_station'] = None
            assign_order_df['assigned_pod'] = None
            assign_order_df['status'] = -3
            assign_order_df.to_csv('assign_order.csv', index=False)
        new_file_df = pd.read_csv(file_path)
                  
        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = new_file_df[(new_file_df['order_arrival']<= current_second) & 
                               (new_file_df['order_arrival'] > previous_second) &
                               (new_file_df['status'] == -3)]
        grouped_orders = new_orders.groupby('order_id')

        for order_id, group in grouped_orders:
            order_items = group[['item_id', 'item_quantity']].to_dict('records')
            order = Order(order_id=order_id, order_arrival=current_second)

            # Add each item in the group to the order
            for item in order_items:
                order.add_sku(item['item_id'], item['item_quantity'])

            self.order_manager.add_order(order)
            # DB
            upsert_order_history(order.order_id, arrival_time=self._tick)

        return new_orders

    def get_movable_objects(self):
        result = []
        for o in self._objects:
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result

    def process_orders(self):
        # Step 1: Robot job initialization
        robots_location = [
            [o.pos_x, o.pos_y] for o in self.get_movable_objects()
            if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle'
            and len(self.job_queue) > 0
        ]
        # Step 2: Trigger preassign logic
        if self.poa_first:
            advanced_table = self.get_advanced_table()
        # Step 3: Assign orders based on conditions
        total_empty_bin = self.get_total_empty_bin()
        if sum(total_empty_bin.values()) >= 1 and self._tick >= 1:
            if self.poa_podmatch:
                self.assign_order_old()
            if self.poa_first:
                self.assign_order()
            if self.poa_second:
                self.xxx()
        # Step 4: Record last order for each station
        if self.poa_first:
            for st in [v for k, v in self.station_manager.stations_by_id.items() if 'picker' in k]:
                self.last_order[st.station_id] = advanced_table.loc[advanced_table['station_id'] == st.station_id, 'order_id'].tolist()
            print(self.last_order)
        # Step 5: Start unfinished orders
        assign_order_df = pd.read_csv('assign_order.csv')
        for order in self.order_manager.unfinished_orders:
            if order.station_id is None:
                continue
            if order.process_start_time <= 0:
                order.start_processing(int(self._tick))
        assign_order_df.to_csv('assign_order.csv', index=False)
        # Step 6: Process PPS logic
        if self.pps_demand or self.pps_pileon:
            for station in filter(lambda s: s.station_type == 'picker' and len(s.incoming_pod) < 11, self.station_manager.stations):
                priority_orders, general_orders = {}, {}
                for order in station.orders:
                    remaining_skus = order.get_remaining_skus()
                    if 0 < len(remaining_skus) <= 2:
                        if self.priority_order:
                            priority_orders[order.order_id] = remaining_skus
                        general_orders[order.order_id] = remaining_skus
                    else:
                        general_orders[order.order_id] = remaining_skus
                print(f"[DEBUG] priority orders {priority_orders}")
                # Handle priority orders first
                if priority_orders:
                    pod_assigned = False
                    for order_id, remaining_skus in priority_orders.items():
                        idle_pods = {pod for pod in self.pod_manager.sku_to_pods.get(list(remaining_skus.keys())[0], []) if self.pod_manager.is_idle(pod.pod_id)}
                        for pod in idle_pods:
                            pod_id = pod.pod_id
                            print(f"[DEBUG] pod_id {pod_id} with")
                            can_fulfill = any(
                                sku in pod.skus and pod.skus[sku]["current_qty"] >= qty
                                for sku, qty in remaining_skus.items()
                            )
                            print(f"[DEBUG] can fulfill {can_fulfill}")
                            if can_fulfill:
                                sku_to_quantity = {sku: qty for sku, qty in remaining_skus.items()}
                                sku_to_order_map = {sku: [(order_id, qty)] for sku, qty in remaining_skus.items()}
                                job = self.add_picking_task_after_pps(station, pod, sku_to_order_map, sku_to_quantity)
                                self.job_queue.append(job)
                                for sku, qty in sku_to_quantity.items():
                                    upsert_job_task(
                                        pod_id=str(pod.pod_id),
                                        order_id=str(order_id),
                                        sku=str(sku),
                                        qty=str(qty),
                                        assigned_station=station.station_id,
                                        pod_assigned_time=self._tick,
                                        status="queue",
                                    )
                                pod_assigned = True
                                break
                        if pod_assigned:
                            break
                    if pod_assigned:
                        continue

                # Process general orders
                sku_to_quantity, sku_to_order_map = defaultdict(int), defaultdict(list)
                for o_id, remaining_skus in general_orders.items():
                    for sku, qty in remaining_skus.items():
                        sku_to_quantity[sku] += qty
                        sku_to_order_map[sku].append((o_id, qty))

                # Stockout Count metric: a requested SKU whose WAREHOUSE stock is zero
                # (all its pods empty, per Lamballais et al. 2018) is a true stockout for
                # this order line. Recorded once per (order_id, sku) so it counts unique
                # affected order lines, not per-tick retries.
                for sku, orders_q in sku_to_order_map.items():
                    sku_meta = self.pod_manager.skus_data.get(sku, {})
                    if float(sku_meta.get('current_global_qty', 1)) <= 0:
                        for o_id, _q in orders_q:
                            self._stockout_orderlines.add((o_id, sku))

                if not sku_to_quantity:
                    print(f"skipping pod search for station {station.station_id}")
                    continue

                # Pod selection
                if self.pps_demand:
                    backlog_skus = defaultdict(int)
                    for o in filter(lambda o: o.station_id is None and o.order_id not in self.order_manager.preassign_order_ids, self.order_manager.unfinished_orders):
                        for sku, q in o.skus.items():
                            backlog_skus[sku] += q["total_quantity"]
                    pod, score = self.find_best_pod(backlog_skus, list(sku_to_quantity.keys()), mode="demand")
                else:
                    pod, score = self.find_best_pod(sku_to_quantity, list(sku_to_quantity.keys()), mode="pile_on")

                if not pod:
                    # No pod can contribute any of the wanted SKUs → those SKUs are
                    # depleted across all idle pods (stockout-induced wait).
                    self._wait_stockout += 1
                    continue

                job = self.add_picking_task_after_pps(station, pod, sku_to_order_map, sku_to_quantity)
                if len(job.orders) > 0:
                    self.job_queue.append(job)
                    for triplet in job.orders:
                        upsert_job_task(
                            pod_id=str(job.pod.pod_id),
                            order_id=str(triplet[0]),
                            sku=str(triplet[1]),
                            qty=str(triplet[2]),
                            assigned_station=station.station_id,
                            pod_assigned_time=self._tick,
                            status="queue",
                        )

    # def process_orders(self):
    #     robots_location = []
    #     for o in self.get_movable_objects():
    #         if len(self.job_queue) > 0:
    #             job: RobotJob = self.job_queue[0]

    #             if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
    #                 robots_location.append([o.pos_x, o.pos_y])

    #     if self.poa_first:
    #         advanced_table = self.get_advanced_table()  # di dalam sini ada proses preassign


    #     # misal kamu mau tau total keseluruhan -> x = sum(self.get_total_empty_bin().values())
    #     total_empty_bin = self.get_total_empty_bin()
    #     if sum(total_empty_bin.values()) >= 1 and self._tick >=1:
    #         if self.poa_podmatch:
    #             self.assign_order_old()
    #         if self.poa_first:
    #             self.assign_order()
    #         if self.poa_second:
    #             self.xxx()

    #     if self.poa_first:
    #         picking_station = [v for k, v in self.station_manager.stations_by_id.items() if 'picker' in k]
    #         for st in picking_station:
    #             self.last_order[st.station_id] = advanced_table.loc[advanced_table['station_id'] == st.station_id, 'order_id'].tolist()
    #         print(self.last_order)
    #     for order in self.order_manager.unfinished_orders:
    #         assign_order_df = pd.read_csv('assign_order.csv')
    #         if order.station_id is None:
    #             continue

    #         # print(f"[DEBUG] order {order.order_id} is not None and keep running the rest")
    #         if order.process_start_time <= 0:
    #             # print(f"[DEBUG] start_process {order.order_id}")
    #             order.start_processing(int(self._tick))
                
    #         assign_order_df.to_csv('assign_order.csv', index=False)

    #     if self.pps_demand:
    #         print("pps_demand")
    #         for station in [st for st in self.station_manager.stations if st.station_type == 'picker']:
    #             print(f"incoming_pod {station.station_id} {station.incoming_pod}")
    #             if len(station.incoming_pod) < 11:
    #                 priority_order = {}
    #                 general_order = {}
    #                 for order in station.orders:
    #                     remaining_skus = order.get_remaining_skus()
    #                     if len(remaining_skus) <= 2:
    #                         # priority_order[order.order_id] = remaining_skus
    #                         general_order[order.order_id] = remaining_skus
    #                     else:
    #                         general_order[order.order_id] = remaining_skus
    #                     # update0617 print(f"order {order.order_id} has remaining skus {remaining_skus}")
    #                 sku_to_quantity = defaultdict(int)
    #                 sku_to_list_order_id_and_quantity = defaultdict(list)
    #                 for o_id, remaining_skus in general_order.items():
    #                     for sku, qty in remaining_skus.items():
    #                         sku_to_quantity[sku] += qty
    #                         sku_to_list_order_id_and_quantity[sku].append((o_id, qty))
    #                 print(f"for station {station.station_id} with sku_to_quantity {sku_to_quantity}")
    #                 print(f"sku in station {station.skus_in_station}")
    #                 if not sku_to_quantity:
    #                     print(f"skipping pod search for station {station.station_id}")
    #                     continue

    #                 ## PPS Demand
    #                 backlog_skus = defaultdict(int)
    #                 unassigned_orders = [order for order in self.order_manager.unfinished_orders if 
    #                                      (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
    #                 for o in unassigned_orders:
    #                     for sku, q in o.skus.items():
    #                         backlog_skus[sku] += q["total_quantity"]
    #                 highest_demand_on_pod, demand_score = self.find_pod_with_the_highest_demand(backlog_skus, list(sku_to_quantity.keys()))
    #                 print("highest_demand_on_pod", highest_demand_on_pod, "demand_score", demand_score)
    #                 if not highest_demand_on_pod:
    #                     print("assign nothing")
    #                     continue

    #                 job = self.add_picking_task_after_pps(
    #                     station,
    #                     highest_demand_on_pod,
    #                     sku_to_list_order_id_and_quantity,
    #                     sku_to_quantity
    #                 )

    #                 if len(job.orders) > 0:
    #                     self.job_queue.append(job)
    #                     write_record_to("record_record.csv", [f"{self._tick:.2f}", 'job_append', job.pod, job.pod.coordinate], ['Time', 'Event', 'Pod ID', 'Location'])



    #     if self.pps_pileon:
    #         print("pps_pileon")
    #         for station in [st for st in self.station_manager.stations if st.station_type == 'picker']:
    #             print(f"incoming_pod {station.station_id} {station.incoming_pod}")
    #             if len(station.incoming_pod) < 11:
    #                 priority_order = {}
    #                 general_order = {}
    #                 for order in station.orders:
    #                     remaining_skus = order.get_remaining_skus()
    #                     if len(remaining_skus) <= 2:
    #                         priority_order[order.order_id] = remaining_skus
    #                         # general_order[order.order_id] = remaining_skus
    #                     else:
    #                         general_order[order.order_id] = remaining_skus
    #                     # update0617 print(f"order {order.order_id} has remaining skus {remaining_skus}")
    #                 # if priority order, then blabla
    #                 # if not
    #                 sku_to_quantity = defaultdict(int)
    #                 sku_to_list_order_id_and_quantity = defaultdict(list)
    #                 # print("general_order", general_order)
    #                 for o_id, remaining_skus in general_order.items():
    #                     for sku, qty in remaining_skus.items():
    #                         sku_to_quantity[sku] += qty
    #                         sku_to_list_order_id_and_quantity[sku].append((o_id, qty))
    #                 print(f"for station {station.station_id} with sku_to_quantity {sku_to_quantity}")
    #                 print(f"sku in station {station.skus_in_station}")
    #                 if not sku_to_quantity:
    #                     print(f"skipping pod search for station {station.station_id}")
    #                     continue
    #                 ## PPS Pile On
    #                 highest_pile_on_pod, pile_on_score = self.find_pod_with_the_highest_pile_on(sku_to_quantity)
    #                 print("highest_pile_on_pod", highest_pile_on_pod, "pile_on_score", pile_on_score)
                    
    #                 ## try to fix teleport
    #                 # highest_pile_on_pod = self.pod_manager.get_pod_by_id(highest_pile_on_pod.pod_id)
    #                 job = self.add_picking_task_after_pps(
    #                     station,
    #                     highest_pile_on_pod,
    #                     sku_to_list_order_id_and_quantity,
    #                     sku_to_quantity
    #                 )

    #                 if len(job.orders) > 0:
    #                     self.job_queue.append(job)
    #                     for triplet in job.orders:
    #                         upsert_job_task(
    #                             pod_id=str(job.pod.pod_id),
    #                             order_id=str(triplet[0]),
    #                             sku=str(triplet[1]),
    #                             qty=str(triplet[2]),
    #                             assigned_station=station.station_id,
    #                             pod_assigned_time=self._tick,
    #                             status="queue",
    #                         )
    #                     write_record_to("record_record.csv", [f"{self._tick:.2f}", 'job_append', job.pod, job.pod.coordinate], ['Time', 'Event', 'Pod ID', 'Location'])


    #                 # TODO: order.commit_quantity (to tell that that order remaining skus are decreased)
    #                 # TODO: pod.pick_sku(sku, quantity_to_take) (reduct the item inside pod)
    #                 # TODO: self.pod_manager.reduct_sku_data(sku, quantity_to_take) (reduct item in global stock list)
    #                 # TODO: station.add_pod(pod.pod_id)
    #                 # TODO: pod.station = station
    #                 # TODO: update assign_order_df or whatever...
    #                 # TODO: self.pod_manager.mark_pod_not_available(pod.coordinate) (set the pod to not idle)
    #                 # TODO: station.reduce_sku_from_station(sku, quantity_to_take) (reduce the remaining skus in station)
    #                 # TODO: job.add_picking_task(order.order_id, sku, quantity_to_take) (the picking task list inside robotjob)
                    
    #                 # TODO: self.job_queue.append(job)
    def find_best_pod(
        self, 
        sku_to_quantity: dict, 
        relevant_skus: list, 
        mode: str = "pile_on"  # or "demand"
    ):  # type: ignore

        # Step 1: Collect pod candidates from relevant skus
        pod_candidates: set[Pod] = set()
        for sku in relevant_skus:
            pod_candidates.update(self.pod_manager.sku_to_pods.get(sku, []))

        # Step 2: Filter only idle pods
        pod_candidates = {pod for pod in pod_candidates if self.pod_manager.is_idle(pod.pod_id)}

        print(f"[DEBUG] Checking candidates for mode={mode} skus={relevant_skus}")
        print(f"[DEBUG] pod_candidates={pod_candidates}")

        if not pod_candidates:
            idle_count = sum(1 for p in self.pod_manager.pods if self.pod_manager.is_idle(p.pod_id))
            total_count = len(self.pod_manager.pods)
            print(f"[DEBUG] pod_candidates empty — idle pods: {idle_count}/{total_count}")
            return None, -1

        # Step 3: Score function
        def score_pod(pod: Pod) -> int:
            score = 0
            for sku, req_qty in sku_to_quantity.items():
                if sku in pod.skus:
                    current_total = pod.skus[sku]['current_qty']
                    score += min(current_total, req_qty)
            return score

        # Step 4: Rank pods by score
        ranked_pods = sorted(
            [(pod, score_pod(pod)) for pod in pod_candidates],
            key=lambda x: x[1],
            reverse=True
        )

        print(f"[DEBUG] ranked_pods (mode={mode}) = {ranked_pods}")
        # Discard pods that can contribute nothing (all relevant SKUs at qty=0)
        if ranked_pods[0][1] <= 0:
            return None, -1
        return ranked_pods[0]

    def add_picking_task_after_pps(self, station: Station, pod: Pod, sku_to_list_order_id_and_quantity: dict, sku_to_quantity: dict):
        latest_pod_location = get_pod_location(pod.pod_id)
        if latest_pod_location:
            pod.pos_x, pod.pos_y = int(latest_pod_location[0]), int(latest_pod_location[1])
        job = RobotJob(pod.coordinate, station_id=station.station_id, pod=pod)
        for sku in sku_to_list_order_id_and_quantity:
            # sort based on the least quantity for each sku
            sku_to_list_order_id_and_quantity[sku] = sorted(sku_to_list_order_id_and_quantity[sku], key=lambda x: x[1])
            if sku in pod.skus:
                if pod.get_quantity(sku) >= sku_to_quantity[sku]:
                    quantity_to_take = sku_to_quantity[sku]
                    # set the order list for job
                else:
                    quantity_to_take = pod.get_quantity(sku)
                    # set the order list for job
                
                tmp = quantity_to_take
                for o_id, qty in sku_to_list_order_id_and_quantity[sku]:
                    if tmp <= 0:
                        break
                    # order.commit_quantity
                    self.order_manager.get_order_by_id(o_id).commit_quantity(sku, min(qty, tmp))
                    # job.add_picking_tas
                    job.add_picking_task(o_id, sku, min(qty, tmp))
                    tmp = tmp - min(qty, tmp)
                
                # pod.pick_sku
                pod.pick_sku(sku, quantity_to_take)
                # self.pod_manager
                self.pod_manager.reduce_sku_data(sku, quantity_to_take)
                # station.reduce_sku
                station.reduce_sku_from_station(sku, quantity_to_take)

        station.add_pod(pod.pod_id)
        pod.station = station
        # print(f"[DEBUG] assign job pod {pod.id} coordinate {pod.coordinate}")
        self.pod_manager.mark_pod_not_available(pod)
        return job

    def find_pod_with_the_highest_pile_on(self, sku_to_quantity: dict) -> (Pod, int): # type: ignore
        # dict of order: {order_id_1: {sku1: X, sku2: Y}, order_id_2: {...}}    
        sku_list = sku_to_quantity.keys()
        pod_candidates: set[Pod] = set()
        for sku in sku_list:
            pod_candidates.update(self.pod_manager.sku_to_pods.get(sku, []))

        print(f"checking candicate for sku {sku_list}")
        print(f"pod_candidates {pod_candidates}")

        def pile_on_score(pod: Pod):
            # if pod.is_idle:
            if self.pod_manager.is_idle(pod.pod_id):
                # print(f"pod {pod} is idle")
                score = 0
                for sku, req_qty in sku_to_quantity.items():
                    if sku in pod.skus:
                        current_total = pod.skus[sku]['current_qty']
                        score += min(current_total, req_qty)  # Only count up to what's needed
            else:
                # print(f"pod {pod} is NOT idle")
                score = -1
            return score

        ranked_pods = sorted(
            [(pod, pile_on_score(pod)) for pod in pod_candidates],
            key=lambda x: x[1],
            reverse=True
        )
        print("ranked_pods", ranked_pods)
        return ranked_pods[0]
    
    def find_pod_with_the_highest_demand(self, sku_to_quantity: dict, station_unfinished_skus: list) -> (Pod, int): # type: ignore
        # dict of order: {order_id_1: {sku1: X, sku2: Y}, order_id_2: {...}}    
        pod_candidates: set[Pod] = set()
        for sku in station_unfinished_skus:
            pod_candidates.update(self.pod_manager.sku_to_pods.get(sku, []))
        # filter the pod status
        # pod_candidates = {po for po in pod_candidates if po.is_idle}
        pod_candidates = {po for po in pod_candidates if self.pod_manager.is_idle(po.pod_id)}
        print(f"checking candidate for sku {station_unfinished_skus}")
        print(f"pod_candidates {pod_candidates}")

        # for early stage, if empty, then assign random ?
        if not pod_candidates:
            return None, -1
            pod_candidates.update([po for po in self.pod_manager.pods if po.is_idle])

        def demand_score(pod: Pod):
            # if pod.is_idle:
            if self.pod_manager.is_idle(pod.pod_id):
                # print(f"pod {pod} is idle")
                score = 0
                for sku, req_qty in sku_to_quantity.items():
                    if sku in pod.skus:
                        current_total = pod.skus[sku]['current_qty']
                        score += min(current_total, req_qty)  # Only count up to what's needed
            else:
                # print(f"pod {pod} is NOT idle")
                score = -1
            return score

        ranked_pods = sorted(
            [(pod, demand_score(pod)) for pod in pod_candidates],
            key=lambda x: x[1],
            reverse=True
        )
        print("ranked_pods", ranked_pods)
        return ranked_pods[0]

    def write_to_csv(self, filename, header, data):
        folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = os.path.join(folder_path, filename)
        file_exists = os.path.exists(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data)

    def get_station_orders_info(self):
        station_orders = []
        for station in sorted(self.station_manager.stations, key=lambda x: x.station_id):
            if station.is_picker_station():
                order_list = ', '.join(map(str, station.order_ids)) if station.order_ids else "Empty"
                station_orders.append(order_list)
        while len(station_orders) < 3:
            station_orders.append("Empty")
        return station_orders

    def generateResult(self):
        result = super().generateResult()
        station_orders = self.get_station_orders_info()
        return [result, station_orders]

    def get_fulfilment_table(self, mode="FS", excludes=[]):
        # MODE: 
        # FS=fully supplied 
        # OTW= only on incoming pod - pod in stations
        # F3=incoming pod - 3 queue

        # station_ids = [station.station_id for station in self.station_manager.stations]
        # order_ids = [order.order_id for order in self.order_manager.unfinished_orders]

        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]

        # Gather all assigned order IDs across all stations
        # assigned_order_ids = {order_id for station in self.station_manager.stations for order_id in station.order_ids}

        # Filter orders whose order_id is NOT in assigned_order_ids
        # unassigned_orders = [order for order in self.order_manager.unfinished_orders if order.order_id not in assigned_order_ids]
        # unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None)]
        ## Activate this only if you use advanced table
        unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
        
        picking_station_ids = [station.station_id for station in picking_stations]
        unassigned_order_ids = [order.order_id for order in unassigned_orders]

        # Initialize fulfillment matrix with zeros
        fulfilment_matrix = pd.DataFrame(0.0, index=unassigned_order_ids, columns=picking_station_ids)

        for station in picking_stations:
            # Build station SKU availability from incoming pods
            station_sku_quantity = {}
            if mode == "FS":
                list_of_pods = station.incoming_pod
            elif mode == "OTW":
                # TODO: incoming pods - is in station
                robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in station.incoming_pod
                          and not o.is_in_station_path()]
                list_of_pods = [o.job.pod.pod_id for o in robots_otw]
            elif mode == "F3":
                # TODO: incoming pods - self.robot_queue_order
                robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in station.incoming_pod
                          and o not in excludes]
                list_of_pods = [o.job.pod.pod_id for o in robots_otw]
            for pod_id in list_of_pods:
                pod = self.pod_manager.get_pod_by_id(pod_id)
                for sku, details in pod.skus.items():
                    if details['current_qty'] > 0:
                        station_sku_quantity[sku] = station_sku_quantity.get(sku, 0) + details['current_qty']

            for order in unassigned_orders:
                total_order_qty = sum([x.get('total_quantity') for x in order.skus.values()])
                fulfilled_qty = 0
                for sku, val in order.skus.items():
                    available_qty = station_sku_quantity.get(sku, 0)
                    fulfilled_qty += min(val.get('total_quantity'), available_qty)

                fulfillment_rate = fulfilled_qty / total_order_qty if total_order_qty > 0 else 0.0
                fulfilment_matrix.at[order.order_id, station.station_id] = fulfillment_rate

        return fulfilment_matrix

    def get_total_empty_bin(self):
        bin_dict = {}
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        for station in picking_stations:
            bin_dict[station.station_id] = station.max_orders - len(station.order_ids)
        return bin_dict
    
    def assign_order_with_advanced_table(self, df):
        empty_bin_dict = self.get_total_empty_bin()
        final_selection = {}
        for picker, count in empty_bin_dict.items():
            # for the order_id list in df, if it's not in station
            # self.last_order[picker]
            # self.station_manager.get_station_by_id(picker).order_ids
            try:
                print(f"count: {count}")
                list_of_just_finished_order = [
                    oid for oid in self.last_order[picker] if oid not in df.loc[df['station_id'] == picker, 'order_id'].tolist()
                ]
                print(f"list_of_just_finished_order: {list_of_just_finished_order}")
                list_of_pre_assigned_order = [
                    self.preassign_dict[k] for k in list_of_just_finished_order
                ]
                print(f"preassign_dict: {self.preassign_dict}")
                print(f"list_of_pre_assigned_order: {list_of_pre_assigned_order}")
                if len(list_of_pre_assigned_order) != count:
                    return self.assign_order()
                for n in list_of_pre_assigned_order:
                    final_selection.setdefault(picker, []).append(n)
            except Exception as e:
                print(f"exception with e: {type(e).__name__} {e}")
                return self.assign_order()
            # candidates = df.loc[df['station_id'] == picker, 'pre_assign'].tolist()
            # candidates = [c for c in candidates if c]
            # if not candidates:
            #     return self.assign_order()
                
            # for n in range(count):
            #     final_selection.setdefault(picker, []).append(candidates[0])
            #     del candidates[0]
        print(f"final_selection: {final_selection}")
        self.put_order_to_picking_station(final_selection)

        return final_selection

    def assign_order(self):
        # TODO: for advanced table version, just use the preassign as the assign
        fulfilment_table = self.get_fulfilment_table(mode="OTW")
        empty_bin_dict = self.get_total_empty_bin()

        print("assign_order is triggered")
        print(fulfilment_table)
        print(empty_bin_dict)

        # Step 1: Flatten all candidate values with their source column
        candidates = []
        for picker, count in empty_bin_dict.items():
            top_rows = fulfilment_table[picker].sort_values(ascending=False).head(count * 3)  # get more to allow fallback if conflict
            for index, value in top_rows.items():
                candidates.append({'index': index, 'picker': picker, 'value': value})

        # Step 2: Sort all candidates by value descending
        candidates = sorted(candidates, key=lambda x: x['value'], reverse=True)

        # Step 3: Pick best combination without duplicate indices
        final_selection = {}
        used_indices = set()

        for candidate in candidates:
            picker = candidate['picker']
            index = candidate['index']

            if index in used_indices:
                continue
            if empty_bin_dict[picker] > 0:
                final_selection.setdefault(picker, []).append(index)
                empty_bin_dict[picker] -= 1
                used_indices.add(index)

            # Stop early if all picks are satisfied
            if all(v == 0 for v in empty_bin_dict.values()):
                break
        
        print("Final selection:", final_selection)

        # Put the decision into action
        self.put_order_to_picking_station(final_selection)

        return final_selection
    
    def assign_order_old(self): # buat yang baseline
        fulfilment_table = self.get_fulfilment_table("FS")
        empty_bin_dict = self.get_total_empty_bin()

        print("assign_order is triggered")
        print(fulfilment_table)
        print(empty_bin_dict)

        # Step 1: Flatten all candidate values with their source column
        candidates = []
        for picker, count in empty_bin_dict.items():
            top_rows = fulfilment_table[picker].sort_values(ascending=False).head(count * 3)  # get more to allow fallback if conflict
            for index, value in top_rows.items():
                candidates.append({'index': index, 'picker': picker, 'value': value})

        # Step 2: Sort all candidates by value descending
        candidates = sorted(candidates, key=lambda x: x['value'], reverse=True)

        # Step 3: Pick best combination without duplicate indices
        final_selection = {}
        used_indices = set()

        for candidate in candidates:
            picker = candidate['picker']
            index = candidate['index']

            if index in used_indices:
                continue
            if empty_bin_dict[picker] > 0:
                final_selection.setdefault(picker, []).append(index)
                empty_bin_dict[picker] -= 1
                used_indices.add(index)

            # Stop early if all picks are satisfied
            if all(v == 0 for v in empty_bin_dict.values()):
                break
        
        print("Final selection:", final_selection)

        # Put the decision into action
        self.put_order_to_picking_station(final_selection)

        return final_selection
        
    def put_order_to_picking_station(self, final_selection):
        assign_order_df = pd.read_csv('assign_order.csv')

        for picker_name, order_ids in final_selection.items():
            for order_id in order_ids:
                order = self.order_manager.get_order_by_id(order_id)
                order.assign_station(picker_name)
                self.station_manager.get_station_by_id(picker_name).add_order(order_id, order)

                assign_order_df.loc[assign_order_df['order_id'] == order.order_id, 'assigned_station'] = picker_name
                assign_order_df.loc[assign_order_df['order_id'] == order.order_id, 'status'] = -1
                # DB
                upsert_order_history(order_id, assigned_station=picker_name, order_assigned_time=self._tick)
            
        assign_order_df.to_csv('assign_order.csv', index=False)

    @staticmethod
    def _calculate_two_coordinates(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def distance_robot_to_station(self, robot: Robot, station: Station):
        return self._calculate_two_coordinates((robot.pos_x, robot.pos_y), (station.coordinate.x, station.coordinate.y))

    def sort_pod_order(self, robots: List[Robot], station: Station):
        print("inside sort_pod_order")
        print(f"robot inside: {robots}")
        others = []
        current = None
        one_right = None
        two_right = None
        one_up = None
        two_up = None
        one, two, three = None, None, None
        station_coordinate = station.coordinate
        for r in robots:
            if r.pos_x == station_coordinate.x and r.pos_y == station_coordinate.y:
                current = r
            elif r.pos_x == station_coordinate.x + 1 and r.pos_y == station_coordinate.y:
                one_right = r
            elif math.floor(r.pos_x) == station_coordinate.x + 2 and r.pos_y == station_coordinate.y:
                two_right = r
            elif r.pos_x == station_coordinate.x and math.floor(r.pos_y) == station_coordinate.y + 1:
                one_up = r
            elif r.pos_x == station_coordinate.x and math.floor(r.pos_y) == station_coordinate.y + 2:
                two_up = r
            else:
                others.append(
                    (r, self.distance_robot_to_station(r, station))
                )
        print(f"total others: {others}")
        # TODO: sort others according to the distance
        if current:
            one = current
            if one_up and not one_right:
                two = one_up
                if two_up:
                    three = two_up
                else:
                    three = others[0][0]
                return (one, two, three)
            elif one_right:
                two = one_right
                if two_right:
                    three = two_right
                else:
                    three = others[0][0]
                return (one, two, three)
            else:
                if two_up:
                    two = two_up
                    three = others[0][0]
                else:
                    two = others[0][0]
                    three = others[1][0]
                return (one, two, three)
        else:
            return (None, None, None)

    def get_advanced_table_only(self):
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        if not self.robot_queue_order:
            self.robot_queue_order = {picker.station_id: [] for picker in picking_stations}

        df_dicts = []
        for picker in picking_stations:
            station_id = picker.station_id
            robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in picker.incoming_pod]
            
            inside_station = []
            currently_picking = None
            for r in robots_otw:
                if r.is_being_process_on_station():
                    currently_picking = r
                
                if r.is_in_station_path():
                    inside_station.append(r)
                    if r.id not in self.robot_queue_order.get(station_id, []):
                        # print(f"adding robot queue {r} to {station_id}")
                        self.robot_queue_order[station_id].append(r.id)
                    print(f"{r} is_in_station_path {station_id}")

            for rid in self.robot_queue_order[station_id]:
                if rid not in [x.id for x in inside_station]:
                    # print(f"removing robot queue {r} from {station_id}")
                    self.robot_queue_order[station_id].remove(rid)
            print(f"current robot inside station {station_id}: {self.robot_queue_order[station_id]}")
            my_robot_queue_order = [None] * len(self.robot_queue_order[station_id])
            for n, rid in enumerate(self.robot_queue_order[station_id]):
                my_robot_queue_order[n] = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == rid]
                my_robot_queue_order[n] = my_robot_queue_order[n][0] if my_robot_queue_order[n] else None
            if len(my_robot_queue_order) >= 3:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = my_robot_queue_order[2]
            elif len(my_robot_queue_order) == 2:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = None
            elif len(my_robot_queue_order) == 1:
                first_queue = my_robot_queue_order[0]
                second_queue = None
                third_queue = None
            else:
                first_queue = None
                second_queue = None
                third_queue = None
            for order in picker.orders:
                order_id = order.order_id
                unpicked_skus = order.get_unpicked_skus()
                df_dicts.append({
                    "station_id": station_id,
                    "order_id": order_id,
                    "unpicked_skus": str({int(k): v for k, v in unpicked_skus.items()}),
                    # "robot_inside_station": self.robot_queue_order[station_id],
                    "pod_1": first_queue,
                    "pod_2": second_queue,
                    "pod_3": third_queue,
                    "occupied_1": first_queue.job.orders if (first_queue and first_queue.job) else None,
                    "occupied_2": second_queue.job.orders if (second_queue and second_queue.job) else None,
                    "occupied_3": third_queue.job.orders if (third_queue and third_queue.job) else None,
                    "next_bin_avail": None,
                    "pre_assign": self.preassign_dict.get(order_id, None)
                })
        df = pd.DataFrame(df_dicts)
        df = self.forcast_next_bin_avail(df)
        return df

    def get_advanced_table(self):
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        if not self.robot_queue_order:
            self.robot_queue_order = {picker.station_id: [] for picker in picking_stations}
        # if not self.currently_picking:
        #     self.currently_picking = {picker.station_id: None for picker in picking_stations}
        # print(f"PICKING STATION {picking_stations[0].station_id}")
        # print(f"ORDER IDS {picking_stations[0].order_ids}")
        # for o in picking_stations[0].orders:
        #     print(f"ORDER {o.order_id} has")
        #     print(f"SKUS {o.skus}")
        #     print(f"GET_REMAINING_SKUS {o.get_remaining_skus()}")

        # print(f"PICKING STATION {picking_stations[0].station_id} {picking_stations[0].coordinate}")
        # print(f"INCOMING POD {picking_stations[0].incoming_pod}")
        # print(f"JOB_RUNNING")
        # robots_otw_picking_station = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.job and o.job.pod.pod_id in picking_stations[0].incoming_pod]
        # for r in robots_otw_picking_station:
        #     print(f"sending {r.job.pod.pod_id} to {r.job.station_id} status {r.current_state}")
        # print(f"CURRENTLY PICKING")
        # currently_picking = [r for r in robots_otw_picking_station if r.is_being_process_on_station()]
        # for r in currently_picking:
        #     print(f"sending {r.job.pod.pod_id} to {r.job.station_id} status {r.current_state} {r.pos_x:.2f},{r.pos_y:.2f}")
        # print(f"JOB_QUEUE")
        # for rj in self.job_queue:
        #     print(rj)

        df_dicts = []
        for picker in picking_stations:
            station_id = picker.station_id
            # if self.currently_picking[station_id]:
            #     print(f">>> CURRENT STATUS in {station_id}<<<")
            #     my_robot = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == self.currently_picking[station_id]]
            #     my_robot = my_robot[0] if my_robot else None
            #     print(f"{my_robot}")
            #     print(f"ID: {self.currently_picking[station_id]}")
            #     print(f"picking delay {my_robot.job.picking_delay}")
            #     print(f"state {my_robot.current_state}")
            robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in picker.incoming_pod]
            
            inside_station = []
            currently_picking = None
            for r in robots_otw:
                if r.is_being_process_on_station():
                    currently_picking = r
                    # self.currently_picking[station_id] = r.id
                    # print(f"{r} is_being_process {station_id}")
                    # print(f"ID: {r.id}")
                    # print(f"picking delay {r.job.picking_delay}")
                    # print(f"state {r.current_state}")
                    # print(f"location: {r.pos_x}, {r.pos_y}")
                    # print(f"station location: {picker.coordinate}")
                
                if r.is_in_station_path():
                    inside_station.append(r)
                    if r.id not in self.robot_queue_order.get(station_id, []):
                        # print(f"adding robot queue {r} to {station_id}")
                        self.robot_queue_order[station_id].append(r.id)
                    print(f"{r} is_in_station_path {station_id}")

            for rid in self.robot_queue_order[station_id]:
                if rid not in [x.id for x in inside_station]:
                    # print(f"removing robot queue {r} from {station_id}")
                    self.robot_queue_order[station_id].remove(rid)
            print(f"current robot inside station {station_id}: {self.robot_queue_order[station_id]}")
            my_robot_queue_order = [None] * len(self.robot_queue_order[station_id])
            for n, rid in enumerate(self.robot_queue_order[station_id]):
                my_robot_queue_order[n] = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == rid]
                my_robot_queue_order[n] = my_robot_queue_order[n][0] if my_robot_queue_order[n] else None
            if len(my_robot_queue_order) >= 3:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = my_robot_queue_order[2]
            elif len(my_robot_queue_order) == 2:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = None
            elif len(my_robot_queue_order) == 1:
                first_queue = my_robot_queue_order[0]
                second_queue = None
                third_queue = None
            else:
                first_queue = None
                second_queue = None
                third_queue = None
            for order in picker.orders:
                order_id = order.order_id
                unpicked_skus = order.get_unpicked_skus()
                df_dicts.append({
                    "station_id": station_id,
                    "order_id": order_id,
                    "unpicked_skus": str({int(k): v for k, v in unpicked_skus.items()}),
                    # "robot_inside_station": self.robot_queue_order[station_id],
                    "pod_1": first_queue,
                    "pod_2": second_queue,
                    "pod_3": third_queue,
                    "occupied_1": first_queue.job.orders if (first_queue and first_queue.job) else None,
                    "occupied_2": second_queue.job.orders if (second_queue and second_queue.job) else None,
                    "occupied_3": third_queue.job.orders if (third_queue and third_queue.job) else None,
                    "next_bin_avail": None,
                    "pre_assign": self.preassign_dict.get(order_id, None)
                })
        df = pd.DataFrame(df_dicts)
        df = self.forcast_next_bin_avail(df)
        df = self.pre_assign_order(df)
        print(df)
        return df

    def forcast_next_bin_avail(self, df):
        if df.empty or 'unpicked_skus' not in df.columns:
            return df
        def is_fulfilled(row):
            required = row['unpicked_skus']
            # Flatten all occupied bins into a list
            all_occupied = []
            for col in ['occupied_1', 'occupied_2', 'occupied_3']:
                val = row.get(col)
                if isinstance(val, list):
                    all_occupied.extend(val)
            
            # Sum quantities by SKU
            available = defaultdict(int)
            for _, sku, qty in all_occupied:
                available[sku] += qty

            # Check if every required SKU has enough quantity
            for sku, req_qty in required.items():
                if available[sku] < req_qty:
                    return False
            return True
        def safe_parse(x):
            if not isinstance(x, str):
                return x
            try:
                return ast.literal_eval(x)
            except Exception:
                import re
                cleaned = re.sub(r'np\.int64\((\d+)\)', r'\1', x)
                cleaned = re.sub(r'np\.float64\(([\d.eE+\-]+)\)', r'\1', cleaned)
                return ast.literal_eval(cleaned)
        df['unpicked_skus'] = df['unpicked_skus'].apply(safe_parse)
        df['next_bin_avail'] = df.apply(is_fulfilled, axis=1)
        return df
    
    def pre_assign_order(self, df):
        """
        For each row where 'next_bin_avail' is True and 'pre_assign' is None,
        call choose_order with (station_id, pod_1, pod_2, pod_3),
        and store the result in 'pre_assign'.
        """
        print("PREASSIGN IS CALLED !!!!!!!")
        mask = (df['next_bin_avail'] == True) & (df['pre_assign'].isna())  # noqa: E712
        print(mask)
        df.loc[mask, 'pre_assign'] = df[mask].apply(
            lambda row: self.choose_order(row['station_id'], row['order_id'], row['pod_1'], row['pod_2'], row['pod_3']),
            axis=1
        )
        return df

    # Example implementation
    def choose_order(self, station_id: str, order_id: int, pod_1: Pod, pod_2: Pod, pod_3: Pod):
        print("CHOOSE ORDER IS CALLED !!!!!")
        df = self.get_fulfilment_table(mode="F3", excludes=[pod_1, pod_2, pod_3])
        print("\n\nfulfillment table during pre-assignment")
        print(df)
        print("\n\n")
        if df.empty:
            print("[PRE-ASSIGN FULFULLMENT TABLE IS INVALID!]")
            print(f"for station {station_id}")
            print(f"for order_id {order_id}")
            unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
            unassigned_order_ids = [order.order_id for order in unassigned_orders]
            print(f"unassigned_order_ids {unassigned_order_ids}")
            print(f"pod queue {pod_1.pod_id}, {pod_2.pod_id}, {pod_3.pod_id}")
        df.sort_values(by=station_id, ascending=False, inplace=True)
        val = df.index[0]
        self.order_manager.preassign_order_ids.append(val)
        self.preassign_dict[order_id] = int(val)
        print(f" VALUE {val} {df.iloc[0]}")
        return df.index[0]

    def xxx(self):
        picking_stations = [s for s in self.station_manager.stations if s.station_type == 'picker']
        # Step 1: Get current empty bin per station
        empty_bins = {
            station.station_id: station.max_orders - len(station.order_ids)
            for station in picking_stations
            if (station.max_orders - len(station.order_ids)) > 0
        }

        if not empty_bins:
            return
        
        order_ids = []
        current_picker = list(empty_bins.keys())[0]
        total_order_ids = empty_bins[current_picker]
        fulfilment_fs = self.get_fulfilment_table(mode="FS")
        if fulfilment_fs.empty:
            return
        advanced_df = self.get_advanced_table_only()
        if advanced_df.empty:
            return
        while self.preassign_per_station[current_picker] and empty_bins[current_picker] > 0:
            order_ids.append(self.preassign_per_station[current_picker].popleft())
            empty_bins[current_picker] -= 1
        if empty_bins[current_picker] == 0:
            # assign everything in order_ids
            self.yyy(current_picker, order_ids)
            return
        
        # exclude the preassigned
        exclude_indices = set()
        for q in self.preassign_per_station.values():
            exclude_indices.update(q)
        fulfilment_fs = fulfilment_fs[~fulfilment_fs.index.isin(exclude_indices)]

        order_candidates = fulfilment_fs[current_picker].sort_values(ascending=False).head(empty_bins[current_picker]*3)
        # print("### ORDER CANDIDATES")
        # print(order_candidates)

        next_bin_counts = (
            advanced_df[advanced_df['next_bin_avail'] == True]  # noqa: E712
            .groupby('station_id')
            .size()
            .to_dict()
        )
        # print("### NEXT BIN COUNTS")
        # print(next_bin_counts)
        next_bin_counts = {k: v for k, v in next_bin_counts.items() if k != current_picker}
        # print(f"after filter {next_bin_counts}")
        if not next_bin_counts:
            order_ids.extend(
                order_candidates.index[:empty_bins[current_picker]]
            )
            # print("### ORDER TO BE ASSIGNED")
            # print(order_ids)
            # assign everything in order_ids
            self.yyy(current_picker, order_ids)
            return
        else:
            print(f"next_bin_counts {next_bin_counts}")
            # raise AssertionError(f"there is next_bin_counts current picker {current_picker} next_bin_counts {next_bin_counts}")
            fulfilment_f3 = self.get_fulfilment_table(mode="F3")
            for idx, val in order_candidates.items():
                best_picker = fulfilment_f3.loc[idx].idxmax()
                best_value = fulfilment_f3.loc[idx].max()
                # if best_picker != current_picker and best_value > val:
                #     raise AssertionError(f"current picker {current_picker} score {val} best_picker {best_picker} score {best_value}")
                if (
                        best_value > val and 
                        best_picker != current_picker and
                        best_picker in next_bin_counts and
                        next_bin_counts[best_picker] > 0
                    ):
                    # preassign
                    print("\n\n\n YESSSSSSS WE HAVE PREASSIGN!!!!! \n\n\n")
                    print("original")
                    print(order_candidates)
                    # with open('preassign_record.txt', 'a') as f:
                    #     f.write(f"[tick {self._tick}]current {current_picker} order {idx} score {val} bestpicker {best_picker} score {best_value}\n")
                    insert_pre_assign(
                        self._tick,
                        current_picker,
                        idx,
                        val,
                        best_picker,
                        best_value
                    )
                    self.preassign_per_station[best_picker].append(idx)
                    next_bin_counts[best_picker] -= 1
                    # raise AssertionError
                else:
                    order_ids.append(idx)
                
                if len(order_ids) >= total_order_ids:
                    # process
                    self.yyy(current_picker, order_ids)
                    return
            self.yyy(current_picker, order_ids)
            return

    def yyy(self, station_id, order_ids):
        self.put_order_to_picking_station({station_id: order_ids})
        return
    
    def update_robot_job_for_new_orders(self, job: RobotJob):
        return
        station: Station = self.station_manager.get_station_by_id(job.station_id)
        orders: list[Order] = station.get_orders_in_station()
        for order in orders:
            remaining_skus = order.get_remaining_skus()
            for sku, qty in remaining_skus.items():
                pod: Pod = job.pod
                if sku in pod.skus:
                    if pod.get_quantity(sku) >= qty:
                        quantity_to_take = qty
                    else:
                        quantity_to_take = pod.get_quantity(sku)
                    order.commit_quantity(sku, quantity_to_take)
                    job.add_picking_task(order.order_id, sku, quantity_to_take)
                    pod.pick_sku(sku, quantity_to_take)
                    self.pod_manager.reduce_sku_data(sku, quantity_to_take)
                    station.reduce_sku_from_station(sku, quantity_to_take)