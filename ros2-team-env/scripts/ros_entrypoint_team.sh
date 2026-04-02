#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

if [ -f /workspaces/ros2_ws/install/setup.bash ]; then
  source /workspaces/ros2_ws/install/setup.bash
fi

exec "$@"
