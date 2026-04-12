### MY IMPLEMENTATION ###

# assume LiDAR readings is in a csv file in data/

import serial
import pandas as pd
import numpy as np
import math
from collections import defaultdict

UNKNOWN = 0.5
FULL = 1
EMPTY = 0

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
    for idx, row in enumerate(raw_lidar_data.itertuples()):
        new_ts = row.timestamp_s
        new_dist = int(row.byte2, ORIGINAL_BASE) + int(row.byte3, ORIGINAL_BASE) * RAW_MULT_CONSTANT
        new_strength = int(row.byte4, ORIGINAL_BASE) + int(row.byte5, ORIGINAL_BASE) * RAW_MULT_CONSTANT
        lidar_data.loc[len(lidar_data)] = [new_ts, new_dist, new_strength, np.nan]

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
    
    lidar_data["x_pos"] = xs
    lidar_data["y_pos"] = ys

    return lidar_data

def create_grid(lidar_data):    
    min_x = int(min(lidar_data["x_pos"])) # round down
    max_x = math.ceil(max(lidar_data["x_pos"])) # round up

    min_y = int(min(lidar_data["y_pos"]))
    max_y = math.ceil(max(lidar_data["y_pos"]))

    resolution = 1

    # all are UNKNOWN squares
    grid = np.full((
        math.ceil((max_x - min_x) / resolution),
        math.ceil((max_y - min_y) / resolution)
    ), UNKNOWN)
    
    full_coords = set()
    # populate our occupancy map with objects, from LiDAR data
    for _, row in lidar_data.iterrows():
        # shfit lidar data coords to fit with our numpy array indexing (minimum coord at 0)
        cur_x = int((row.x_pos - min_x) / resolution)
        cur_y = int((row.y_pos - min_y) / resolution)
        grid[cur_x][cur_y] = FULL
        full_coords.add((cur_x, cur_y))


    return grid, full_coords

def in_bounds(x, y, grid):
    return x < len(grid) and x >= 0 and y < len(grid[0]) and y >= 0 

def bresenham(grid, full_coords):
    # the "center" (where our sensor is) is at the center of the bottom edge of the grid
    # remember, it's a 180 degree sensor
    c_x = int(len(grid) / 2)
    c_y = 0
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
        
        
    return grid

if __name__ == '__main__':
    SERVO_PATH = 'data/servo_log.csv'
    LIDAR_PATH = 'data/tf_luna_raw.csv'
    lidar_data = create_df(SERVO_PATH, LIDAR_PATH)
    lidar_data = polar_to_cartesian(lidar_data)
    occupancy_map, full_coords = create_grid(lidar_data)
    occupancy_map = bresenham(occupancy_map, full_coords)
    
    print_o_map(occupancy_map)
    # TODO - write visualization function