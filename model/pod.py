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

    def check_pod_index(self, flagged_skus, kl: float = 0.5, cap: int = 8) -> bool:
        """Layer 2 of Warehouse Inventory-SKU in Pod baseline (Chou et al.).

        Q_p = (Σ_{i∈flagged} U_ip) / |n_p|,  where U_ip = current_qty/limit_qty
        is the pod-level inventory utilization ratio of flagged SKU i, and |n_p|
        is the number of SKUs in the pod, capped at `cap` (=8, or fewer if the pod
        has fewer unique SKUs) so that pods with many at-risk SKUs are not
        overlooked. R_p = 1 iff Q_p >= KL. Only SKUs flagged by Layer 1 are summed.
        """
        if not self.skus or not flagged_skus:
            return False
        n_p = len(self.skus)
        denom = min(n_p, cap) if cap else n_p
        if denom <= 0:
            return False
        sum_u_ip = 0.0
        for sku in flagged_skus:
            d = self.skus.get(sku)
            if d is not None and d['limit_qty'] > 0:
                sum_u_ip += d['current_qty'] / d['limit_qty']
        q_p = sum_u_ip / denom
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