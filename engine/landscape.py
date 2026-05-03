from datetime import datetime

class Landscape:

    def __init__(self, dimension):
        self.dimension = dimension
        self.total_objects = 0
        self._map = []
        self._objects = {}
        self.current_date_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        for i in range(self.dimension+1):
            one_row = []
            for j in range(self.dimension+1):
                one_row.append([])
            self._map.append(one_row)
    
    def get_robot_object(self):
        return self._objects
    
    def _setObjectNew(self, label, x, y, speed, acceleration, heading, state, load_mass):
        self.total_objects += 1

        movement = 'vertical'
        if heading == 270 or heading == 90:
            movement = 'horizontal'

        self._objects[label] = {
            'label': label,
            'x': x,
            'y': y,
            'velocity': speed,
            'acceleration': acceleration,
            'heading': heading,
            'movement': movement,
            'state': state,
            'load_mass': load_mass,
        }

        self._map[round(x)][round(y)].append(self._objects[label])

    def setObject(self, label, x, y, speed, acceleration, heading, state, load_mass):
        rx, ry = max(0, min(self.dimension, round(x))), max(0, min(self.dimension, round(y)))
        if label not in self._objects:
            return self._setObjectNew(label, rx, ry, speed, acceleration, heading, state, load_mass)

        old_x = round(self._objects[label]['x'])
        old_y = round(self._objects[label]['y'])
        old_x = max(0, min(self.dimension, old_x))
        old_y = max(0, min(self.dimension, old_y))

        # check if x or y has changed
        if rx != old_x or ry != old_y:
            # remove from old position
            to_iter = self._map[old_x][old_y]
            for index, e in enumerate(to_iter):
                if e['label'] == label:
                    del to_iter[index]
                    break

            # add to new position
            self._map[rx][ry].append(self._objects[label])

        movement = 'vertical'
        if heading == 270 or heading == 90:
            movement = 'horizontal'

        self._objects[label] = {
            'label': label,
            'x': x,
            'y': y,
            'velocity': speed,
            'acceleration': acceleration,
            'heading': heading,
            'movement': movement,
            'state': state,
            'load_mass': load_mass,
        }

    def getNeighborObject(self, x, y, radius):
        i = x-radius
        j = y+radius
        check = 2*radius+1
        points_to_check = []
        result = []
        while i < x+check:
            j = y+radius
            while j > y-check:
                if i >= 0 and j >= 0 and i <= self.dimension and j <= self.dimension:
                    if i != x or j != y:
                        points_to_check.append([i, j])
                j -= 1
            i += 1

        for p in points_to_check:
            s = self._map[p[0]][p[1]]
            if len(s) > 0:
                for obj in s:
                    result.append(self._objects[obj['label']])

        return result

    def get_neighbor_object(self, x, y):
        s = self._map[round(x)][round(y)]
        if len(s) > 0:
            for obj in s:
                return self._objects[obj['label']]
        return None

    @property
    def objects(self):
        return self._objects

        