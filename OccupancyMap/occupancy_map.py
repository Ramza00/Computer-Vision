import math


class OccupancyMap:
    def __init__(self, resolution: float) -> None:
        # Resolution implies the size of each cell in the occupancy grid.
        self.resolution = resolution
        # Occupied positions are stored as packed 64-bit keys.
        self.occupied_positions: set[int] = set()
        self.x_min = 2**31 - 1
        self.x_max = -(2**31)
        self.y_min = 2**31 - 1
        self.y_max = -(2**31)

    def _to_key(self, x: int, y: int) -> int:
        return (x << 32) | (y & 0xFFFFFFFF)

    def update(self, start_x: float, start_y: float, angle: float, distance: float) -> None:
        # start_x/start_y are in meters, angle is in radians, and distance is in meters.
        # This method removes cells in the sensor path and marks the end cell occupied.
        distance = math.floor(distance / self.resolution)
        start_x = math.floor(start_x / self.resolution)
        start_y = math.floor(start_y / self.resolution)

        # distance/start_x/start_y are now in units of resolution cells.
        end_x = start_x + distance * math.cos(angle)
        end_y = start_y + distance * math.sin(angle)

        self.occupied_positions.add(self._to_key(math.floor(end_x), math.floor(end_y)))

        if end_x < self.x_min:
            self.x_min = math.floor(end_x)
        if end_x > self.x_max:
            self.x_max = math.floor(end_x)
        if end_y < self.y_min:
            self.y_min = math.floor(end_y)
        if end_y > self.y_max:
            self.y_max = math.floor(end_y)

        temp = start_x if start_x < end_x else end_x
        end_x = end_x if end_x >= start_x else start_x
        start_x = temp + 1

        temp = start_y if start_y < end_y else end_y
        end_y = end_y if end_y >= start_y else start_y
        start_y = temp + 1

        length = int(end_x - start_x) if (end_x - start_x > end_y - start_y) else int(end_y - start_y)
        if length <= 1:
            # If the start and end positions are the same, no need to update the map.
            return

        length -= 1
        x_unit = (end_x - start_x) / length
        y_unit = (end_y - start_y) / length

        for i in range(0, length - 1):
            x = start_x + i * x_unit
            y = start_y + i * y_unit
            self.occupied_positions.discard(self._to_key(math.floor(x), math.floor(y)))

    def is_occupied(self, x: int, y: int) -> bool:
        return self._to_key(x, y) in self.occupied_positions

    def get_occupancy_grid(self) -> list[list[bool]]:
        width = self.x_max - self.x_min + 1
        height = self.y_max - self.y_min + 1
        grid = [[False for _ in range(height)] for _ in range(width)]

        for key in self.occupied_positions:
            x = key >> 32
            y = key & 0xFFFFFFFF
            # Convert low 32 bits back to signed int.
            if y >= 0x80000000:
                y -= 0x100000000
            grid[x - self.x_min][y - self.y_min] = True

        return grid
