FROM dustynv/pytorch:2.6-r36.4.0-cu128-24.04


RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe

RUN apt-get update && apt-get install -y \
    gcc \
    git \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    build-essential \
    python3-venv \
    software-properties-common \
    libgtk-3-0t64 \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -U colcon-common-extensions
# Set up the ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get upgrade -y

RUN apt-get update && apt-get install -y ros-jazzy-ros-base python3-rosdep && rm -rf /var/lib/apt/lists/*


RUN rosdep init && rosdep update

ENV ROS_DISTRO=jazzy
ENV ROS_ROOT=/opt/ros/$ROS_DISTRO

# Create ros2_ws and copy files
WORKDIR /root/ros2_ws
SHELL ["/bin/bash", "-c"]

# Install extra dependencies
RUN apt-get update \
    && apt-get -y --quiet --no-install-recommends install \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-compressed-image-transport \
    ros-jazzy-image-publisher

COPY . /root/ros2_ws/src
RUN pip3 install -r src/requirements.txt --break-system-packages
RUN rosdep install --from-paths src --ignore-src -r -y

RUN source /opt/ros/jazzy/setup.bash && colcon build

# Source the ROS2 setup file
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash"]
