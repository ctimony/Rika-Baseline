from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate


class Pod(Object):
    def __init__(self, pod_id: int):
        self.pod_id = pod_id
        self.shape = 'full square'
        # self.shape = 'circle'
        self.object_type = 'pod'
        # self.coordinate = NetLogoCoordinate()
        self.skus = {}
        self.is_idle = True
        self.station = None
        self.need_replenishment = False
        self.reserved_fill = {}  # v7 anti-redundancy: per-SKU units this pod has
        # promised to restock while in transit to replenishment (released on finish)
        self.mass = 0
        self.initial_mass = 0
        self.last_trigger = None
        self.last_opp_score = None
        self.velocity = 0
        self.acceleration = 0

        super().__init__()

    def __eq__(self, other):
        if isinstance(other, Pod):
            return self.pod_id == other.pod_id
        return False

    def __hash__(self):
        # return hash(self.pod_id)
        return hash(getattr(self, "pod_id", id(self)))

    def __repr__(self):
        return f"Pod({self.pod_id})"

    @property
    def coordinate(self):
        return NetLogoCoordinate(self.pos_x, self.pos_y)

    def freeze_initial_mass(self):
        """Call once after all add_sku() to lock initial_mass."""
        self.initial_mass = self.mass

    def add_sku(self, sku, limit_qty, current_qty, threshold, weight, rop_per_slot=0):
        """Add a SKU's slot to this pod. ACCUMULATES across multiple slots of the
        SAME SKU in the SAME pod (stacked storage scheme): pods.csv has one row per
        slot, so a SKU occupying k slots calls add_sku k times — its limit_qty and
        current_qty must SUM over those slots, not be overwritten. (Overwriting kept
        only one slot's worth, silently discarding the rest of the SKU's capacity.)
        rop_per_slot/threshold are per-SKU values, kept from the first slot, not
        summed. n_slots_in_pod counts how many slots of this SKU live in THIS pod
        (incremented once per call), so the per-slot stock can be derived as
        current_qty / n_slots_in_pod (slots of one SKU fill uniformly)."""
        if sku in self.skus:
            self.skus[sku]['limit_qty']   += limit_qty
            self.skus[sku]['current_qty'] += current_qty
            self.skus[sku]['n_slots_in_pod'] += 1
        else:
            self.skus[sku] = {
                'limit_qty': limit_qty,
                'current_qty': current_qty,
                'threshold': threshold,
                'weight': weight,
                'rop_per_slot': rop_per_slot,
                'n_slots_in_pod': 1,
            }
        self.mass += (weight * current_qty)

    @staticmethod
    def slot_stock(details) -> float:
        """Uniform per-slot stock of a SKU in this pod: total current_qty spread
        evenly over its slots in this pod (current_qty / n_slots_in_pod). Slots of
        one SKU fill uniformly, so every slot holds this amount."""
        n = details.get('n_slots_in_pod', 1) or 1
        return float(details.get('current_qty', 0)) / n

    @staticmethod
    def is_slot_low(details, rop_multiplier=1.0) -> bool:
        """True iff this SKU has at least one slot at/under its per-slot reorder
        point (per-slot stock <= rop_per_slot, with rop_per_slot > 0). Because slots
        fill uniformly, 'any slot low' == 'per-slot stock low'."""
        rps = float(details.get('rop_per_slot', 0)) * rop_multiplier
        return rps > 0 and Pod.slot_stock(details) <= rps

    def check_replenishment_needed(self, rop_multiplier=1.0):
        """Check if 50% or more SKUs have a slot at/under their per-slot reorder
        point, to determine if the pod needs to move to a replenishment station."""
        count_below_threshold = 0
        total_skus = len(self.skus)
        alpha = total_skus / 2
        for details in self.skus.values():
            if self.is_slot_low(details, rop_multiplier):
                count_below_threshold += 1

        if count_below_threshold >= alpha:
            return True
        return False

    def check_pod_index(self, flagged_skus, kl: float = 0.2) -> bool:
        """Layer 2 of Warehouse Inventory-SKU in Pod baseline (Chou et al.; Tracy
        Eq 3-5, Hsiao).

        Q_p = (Σ_{i∈flagged} W_i) / n_p,  where W_i = 1 for each SKU flagged by
        Layer 1 (BINARY count of at-risk SKUs in this pod, matching the published
        baseline Q_j = ΣW_i/n_j — NOT the continuous utilisation ratio), and n_p is
        the number of SKUs in the pod. R_p = 1 iff Q_p >= KL. Q_p is therefore the
        TRUE fraction of the pod's SKUs that are flagged (critical), exactly as in
        Tracy/Hsiao. With n_p = 20 SKUs per pod and KL = 0.2, the pod is dispatched
        once at least 4 of its 20 SKUs (20%) are critical — the same trigger point as
        the earlier capped-denominator form (Σ W_i / 8 >= 0.4), but with the true pod
        SKU count as the denominator so Q_p reads as a genuine fraction.
        """
        if not self.skus or not flagged_skus:
            return False
        n_p = min(len(self.skus), 8)   # denominator capped at 8 (SKU-per-pod range)
        if n_p <= 0:
            return False
        # W_i binary: count flagged SKUs actually present in this pod (each = 1).
        w_sum = sum(1 for sku in flagged_skus if sku in self.skus)
        q_p = w_sum / n_p
        return q_p >= kl

    def replenish_all_skus(self):
        """Replenish all SKUs by setting each SKU's current quantity to its limit quantity."""
        for sku in self.skus:
            self.skus[sku]['current_qty'] = self.skus[sku]['limit_qty']
        self.mass = sum(d['weight'] * d['current_qty'] for d in self.skus.values())

    def replenish_skus(self, sku_ids):
        """Replenish ONLY the given SKUs to their limit quantity (others untouched).
        Used by the baseline stockout override so a reactive trip refills just the
        stocked-out SKU rather than freebie-topping-up the whole pod."""
        for sku in sku_ids:
            if sku in self.skus:
                self.skus[sku]['current_qty'] = self.skus[sku]['limit_qty']
        self.mass = sum(d['weight'] * d['current_qty'] for d in self.skus.values())

    def replenish_skus_fillrate(self, sku_ids, fill_rate: float = 1.0):
        """Replenish ONLY the given (critical) SKUs up to fill_rate × limit_qty.
        Others untouched. fill_rate=1.0 fills to full; fill_rate<1 leaves the slot
        partly filled → lower pod mass → lower travel energy (prof's fill-rate lever).
        Filling ONLY the triggering SKUs (not the whole pod) avoids 'free-riding'
        replenishment of non-critical SKUs, so each SKU is restocked only when it is
        itself critical (cleaner, no cross-SKU contamination). Never removes stock."""
        added = {}
        for sku in sku_ids:
            if sku not in self.skus:
                continue
            target = int(round(fill_rate * self.skus[sku]['limit_qty']))
            cur = self.skus[sku]['current_qty']
            if cur < target:
                added[sku] = target - cur
                self.skus[sku]['current_qty'] = target
        self.mass = sum(d['weight'] * d['current_qty'] for d in self.skus.values())
        return added

    def replenish_pod_capacity(self, critical_skus, fill_rate: float = 1.0):
        """Fill the pod up to fill_rate × TOTAL pod capacity (Σ limit_qty), CRITICAL
        SKUs prioritised. Unlike replenish_skus_fillrate (which fills each slot to
        fill_rate × its own limit), this targets the WHOLE-POD mass: critical SKUs are
        topped to full first (so service is protected), then the remaining budget is
        spread over non-critical SKUs until exhausted. fill_rate<1 leaves the pod
        genuinely lighter (lower Σ current → lower mass → lower travel energy) — a
        SHARP energy lever, because it caps total pod mass directly rather than shaving
        a little off every slot. Items weigh 1 unit each, so units == mass == capacity.
        Never removes stock. Returns dict of units added per SKU."""
        budget = fill_rate * sum(d['limit_qty'] for d in self.skus.values())
        added = {}
        critical = set(critical_skus or [])
        # Pass 1 — critical SKUs to full (priority; protects service).
        for sku in critical:
            d = self.skus.get(sku)
            if d is None:
                continue
            room = d['limit_qty'] - d['current_qty']
            if room > 0:
                added[sku] = room
                d['current_qty'] = d['limit_qty']
        used = sum(d['current_qty'] for d in self.skus.values())
        remaining = budget - used
        # Pass 2 — spread remaining budget over non-critical SKUs (in pod order).
        if remaining > 0:
            for sku, d in self.skus.items():
                if remaining <= 0:
                    break
                if sku in critical:
                    continue
                room = d['limit_qty'] - d['current_qty']
                if room <= 0:
                    continue
                give = int(min(room, remaining))
                if give > 0:
                    added[sku] = added.get(sku, 0) + give
                    d['current_qty'] += give
                    remaining -= give
        self.mass = sum(d['weight'] * d['current_qty'] for d in self.skus.values())
        return added

    def pick_sku(self, sku, qty):
        self.skus[sku]['current_qty'] -= qty
        self.mass -= (self.skus[sku]['weight'] * qty)

    def get_quantity(self, sku):
        return self.skus[sku]['current_qty']

    def get_unassigned_skus(self):
        """Return a list of SKUs that have not yet been assigned a pod."""
        unassigned_skus = [sku for sku, details in self.skus.items() if details['pod'] is None]
        return unassigned_skus

    def set_pod_station(self, station):
        self.station = station
        return
    
    def remove_pod_station(self):
        self.station = None
        return

    def get_skus_in_pod(self) -> dict:
        return self.skus