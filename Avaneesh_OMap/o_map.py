import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

UNKNOWN = 0.5
FULL = 1
EMPTY = 0

RESOLUTION = 0.1

LIDAR_PATH = 'data/lidar_spin_scan.csv'


def print_o_map(occupancy_map):
    for row in occupancy_map:
        print(row)

def create_df(servo_path, lidar_path):
    PRECISION = 3
    ORIGINAL_BASE = 16
    RAW_MULT_CONSTANT = 256

    servo_log = pd.read_csv(servo_path)
    raw_lidar_data = pd.read_csv(lidar_path)
    
    # round all timestamps in original df
    for idx, ts in enumerate(raw_lidar_data["timestamp_s"]):
        raw_lidar_data.loc[idx, "timestamp_s"] = round(ts, PRECISION)

    lidar_data = pd.DataFrame(columns = ["timestamp_s", "avg_dist", "avg_strength", "servo_angle_deg"])

    # go through each row in original df
    # transfer the content in the original df to the new df, while interpreting the values in the new df

    updated_lidar_data = {}
    for idx, row in enumerate(raw_lidar_data.itertuples()):
        new_ts = row.timestamp_s
        new_dist = int(row.byte2, ORIGINAL_BASE) + int(row.byte3, ORIGINAL_BASE) * RAW_MULT_CONSTANT
        new_strength = int(row.byte4, ORIGINAL_BASE) + int(row.byte5, ORIGINAL_BASE) * RAW_MULT_CONSTANT
        # lidar_data.loc[len(lidar_data)] = [new_ts, new_dist, new_strength, np.nan]
        updated_lidar_data[len(lidar_data)] = [new_ts, new_dist, new_strength, np.nan]
    lidar_data = pd.DataFrame(updated_lidar_data)
    lidar_data = lidar_data.groupby('timestamp_s').mean()

    for idx, row in enumerate(servo_log.itertuples()):
        cur_ts = round(row.timestamp_s, PRECISION)

        # get row in lidar_data with this ts
        lidar_data.loc[cur_ts, "servo_angle_deg"] = row.servo_angle_deg

    # fill gaps between timestamps for servo_angles
    lidar_data['servo_angle_deg'] = lidar_data['servo_angle_deg'].ffill()

    # ignore NaN values for distance
    return lidar_data.dropna(subset = ["avg_dist"])

def polar_to_cartesian(lidar_data):
    xs = []
    ys = []
    for row in lidar_data.itertuples():
        angle_deg = row.servo_angle_deg
        angle_rad = angle_deg * math.pi / 180

        xs.append(row.avg_dist * np.cos(angle_rad))
        ys.append(row.avg_dist * np.sin(angle_rad))
   

    # ground sensor at (0, 0)
    xs += [0]
    ys += [0]

    return xs, ys

def create_grid(xs, ys):    
    min_x = int(min(xs)) # round down
    max_x = math.ceil(max(xs)) # round up

    min_y = int(min(ys))
    max_y = math.ceil(max(ys))

    # all are UNKNOWN squares
    grid = np.full((
        math.ceil((max_x - min_x) / RESOLUTION),
        math.ceil((max_y - min_y) / RESOLUTION)
    ), UNKNOWN)

    full_coords = set()
    # populate our occupancy map with objects, from LiDAR data
    for x, y in zip(xs, ys):
        if (x == 0 and y == 0):
            continue

        # shfit lidar data coords to fit with our numpy array indexing (minimum coord at 0)
        cur_x = int((x - min_x) / RESOLUTION)
        cur_y = int((y - min_y) / RESOLUTION)
        full_coords.add((cur_x, cur_y))

    return grid, full_coords, min_x, min_y



def in_bounds(x, y, grid):
    return x < len(grid) and x >= 0 and y < len(grid[0]) and y >= 0



def bresenham(grid, full_coords, c_x, c_y):
    # our sensor is at the center of the grid
    for pair in full_coords:
        x, y = pair
        dx = abs(x - c_x)
        dy = abs(y - c_y)
        step_x = 1
        if (x < c_x):
            step_x = -1
        step_y = 1
        if (y < c_y):
            step_y = -1      

        err = dx - dy
        cur_x = c_x
        cur_y = c_y

        # goal of the loop: minimize err, since this represents the ideal line
        while (not (cur_x == x and cur_y == y) and in_bounds(cur_x, cur_y, grid)):
            grid[cur_x][cur_y] = EMPTY
            temp_err = 2 * err

            # adjust x
            # If the error is too low (below -dy), we've moved in the x direction too many times,
            # and we need to balance it by moving in the y direction

            if (temp_err > -dy):
                cur_x += step_x
                err -= dy          

            # adjust y
            if (temp_err < dx):
                cur_y += step_y
                err += dx
    
    # set FULL
    for x, y in full_coords:
        if in_bounds(x, y, grid):
            grid[x][y] = FULL
    return grid



def visualize(data):
    plt.imshow(data, origin='lower', cmap='viridis')
    plt.colorbar() # Adds a legend for the color scale
    plt.show()

def generate():
    # SERVO_PATH = 'data/servo_log.csv'
    # LIDAR_PATH = 'data/tf_luna_raw.csv'
    # lidar_data = create_df(SERVO_PATH, LIDAR_PATH)
    # xs, ys = polar_to_cartesian(lidar_data)
    # occupancy_map, full_coords = create_grid(xs, ys)
    # occupancy_map = bresenham(occupancy_map, full_coords, 0 - int(min(xs)), 0 - int(min(ys)))   
    # visualize(occupancy_map.T)

    
    lidar_data = pd.read_csv(LIDAR_PATH)
    lidar_data.rename(columns={'angle_deg': 'servo_angle_deg', 'distance_m': 'avg_dist'}, inplace=True)
    xs, ys = polar_to_cartesian(lidar_data)
    occupancy_map, full_coords, min_x, min_y = create_grid(xs, ys)
    
    lidar_x = int((0 - min_x) / RESOLUTION)
    lidar_y = int((0 - min_y) / RESOLUTION)
    
    print(f"LIDAR is at {(lidar_x, lidar_y)}")
    return bresenham(occupancy_map, full_coords, lidar_x, lidar_y).T
    

if __name__ == '__main__':
    occupancy_map = generate()
    visualize(occupancy_map)
    