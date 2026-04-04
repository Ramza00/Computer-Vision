Create this structure next to the compose file:

workspace/
  src/

Then inside the container:

mkdir -p src
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
