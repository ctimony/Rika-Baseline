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

    def add_sku(self, sku, limit_qty, current_qty, threshold, weight, rop_per_pod=0):
        """Add a new SKU with its limit, current quantity, and threshold."""
        self.skus[sku] = {
            'limit_qty': limit_qty,
            'current_qty': current_qty,
            'threshold': threshold,
            'weight': weight,
            'rop_per_pod': rop_per_pod,
        }
        self.mass += (self.skus[sku]['weight'] * self.skus[sku]['current_qty'])

    def check_replenishment_needed(self, rop_multiplier=1.0):
        """Check if 50% or more SKUs are below their threshold to determine if the pod needs to move to a
        replenishment station."""
        count_below_threshold = 0
        total_skus = len(self.skus)
        alpha = total_skus / 2
        for details in self.skus.values():
            if details['current_qty'] <= details['rop_per_pod'] * rop_multiplier:
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

    def replenish_all_skus_capped(self, wmax: float):
        """Replenish SKUs up to limit_qty but stop if cumulative pod mass would exceed wmax.
        Returns dict {sku: qty_added} for caller to update global inventory tracker."""
        remaining_capacity = wmax - self.mass
        added = {}
        for sku in sorted(self.skus, key=lambda s: self.skus[s]['weight'], reverse=True):
            d = self.skus[sku]
            needed = d['limit_qty'] - d['current_qty']
            if needed <= 0:
                continue
            addable = int(min(needed, remaining_capacity // d['weight'])) if d['weight'] > 0 else needed
            if addable > 0:
                d['current_qty'] += addable
                remaining_capacity -= addable * d['weight']
                added[sku] = addable
            if remaining_capacity <= 0:
                break
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