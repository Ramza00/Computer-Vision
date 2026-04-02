# ROS 2 Team Docker Environment

This starter kit gives your team a shared ROS 2 Jazzy development environment using Docker.

## What it includes
- ROS 2 Jazzy on Ubuntu 24.04 via the official ROS Docker image
- A non-root `ros` user mapped to the host UID/GID
- `colcon`, `rosdep`, `vcstool`, and common dev tools preinstalled
- Host networking for DDS discovery
- VS Code devcontainer support
- Workspace mounted from `./workspace`

## Why this setup
ROS 2 Jazzy is the current LTS and has Tier 1 support on Ubuntu 24.04 for amd64 and arm64. Official Docker tags include `jazzy`, `jazzy-ros-base`, and `jazzy-ros-core`. Host networking is commonly used for ROS 2 container development because DDS discovery is sensitive to Docker NAT/network isolation. citeturn130874search3turn130874search7turn130874search19turn130874search20

## Prerequisites
- Docker Engine
- Docker Compose plugin
- Linux host recommended for direct ROS 2 networking and GUI forwarding

## Quick start
1. Copy the env template:
   ```bash
   cp .env.example .env
   ```
2. Create the workspace:
   ```bash
   mkdir -p workspace/src
   ```
3. Build the image:
   ```bash
   docker compose build
   ```
4. Start a shell in the container:
   ```bash
   docker compose run --rm ros2-dev
   ```
5. Inside the container, install dependencies and build:
   ```bash
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   colcon build --symlink-install
   source install/setup.bash
   ```

## Team workflow
- Commit source into `workspace/src`
- Keep machine-specific values in `.env`
- Use the same `ROS_DOMAIN_ID` across the team when you want machines to discover each other
- Use different `ROS_DOMAIN_ID`s when multiple teams share the same network

## Common commands
Open shell:
```bash
docker compose run --rm ros2-dev
```

Rebuild after Dockerfile changes:
```bash
docker compose build --no-cache
```

Run a node:
```bash
docker compose run --rm ros2-dev ros2 run demo_nodes_cpp talker
```

## GUI apps
If you need RViz or rqt later, switch the base image from `ros:jazzy-ros-base` to `ros:jazzy-perception` or install the GUI packages you need. The official images publish those tags. citeturn130874search7

## Notes
- This is optimized for development, not hardened production deployment.
- `network_mode: host` works best on Linux. On macOS/Windows, ROS 2 multicast/discovery is more brittle in Docker.
- The compose file mounts `/dev` and uses `privileged: true` so hardware access is straightforward for robotics development.

## Optional hardening later
- Remove `privileged: true` and map only required devices
- Add `cyclonedds` or a discovery-server-based setup for multi-machine deployments
- Add CI build/test jobs for `colcon test`
