FROM ros:jazzy-ros-base
# LABEL org.opencontainers.image.source=https://github.com/LonghornNeurotech/Computer-Vision

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    cmake \
    curl \
    gdb \
    git \
    iputils-ping \
    less \
    lsb-release \
    nano \
    net-tools \
    openssh-client \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    python3-argcomplete \
    ros-dev-tools \
    software-properties-common \
    sudo \
    tmux \
    vim \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN if ! getent group ${USER_GID} >/dev/null; then groupadd --gid ${USER_GID} ${USERNAME}; fi \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
 && usermod -aG sudo ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN rosdep init 2>/dev/null || true

COPY scripts/ros_entrypoint_team.sh /ros_entrypoint_team.sh
RUN chmod +x /ros_entrypoint_team.sh

USER ${USERNAME}
WORKDIR /workspaces/ros2_ws

RUN echo 'source /opt/ros/jazzy/setup.bash' >> /home/${USERNAME}/.bashrc \
 && echo 'if [ -f /install/setup.bash ]; then source /install/setup.bash; fi' >> /home/${USERNAME}/.bashrc \
 && echo 'source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash' >> /home/${USERNAME}/.bashrc \
 && echo 'export ROS_LOCALHOST_ONLY=0' >> /home/${USERNAME}/.bashrc \
 && echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' >> /home/${USERNAME}/.bashrc

ENTRYPOINT ["/ros_entrypoint_team.sh"]
CMD ["bash"]
