FROM dustynv/pytorch:2.6-r36.4.0-cu128
ENV ROS_DISTRO=iron
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}

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
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -U colcon-common-extensions
# Set up the ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get upgrade -y

RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-ros-base python3-rosdep && rm -rf /var/lib/apt/lists/*


RUN rosdep init && rosdep update

# Create ros2_ws and copy files
WORKDIR /root/ros2_ws
SHELL ["/bin/bash", "-c"]

# Install extra dependencies
RUN apt-get update \
    && apt-get -y --quiet --no-install-recommends install \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-compressed-image-transport \
    ros-${ROS_DISTRO}-image-publisher
# #TENSORRT 
ARG TENSORRT_URL="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz"
ARG WGET_FLAGS="--quiet --show-progress --progress=bar:force:noscroll --no-check-certificate"
RUN set -ex && \
    echo "Downloading ${TENSORRT_URL}" && \
    mkdir -p /tmp/tensorrt && \
    cd /tmp/tensorrt && \
    wget $WGET_FLAGS ${TENSORRT_URL} -O TensorRT.tar && \
    tar -xvf TensorRT.tar -C /usr/src && \
    mv /usr/src/TensorRT-* /usr/src/tensorrt

RUN cd /tmp/tensorrt && \
    cp -r /usr/src/tensorrt/lib/* /usr/lib/$(uname -m)-linux-gnu/ && \
    cp -r /usr/src/tensorrt/include/* /usr/include/$(uname -m)-linux-gnu/ && \
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")') && \
    pip3 install /usr/src/tensorrt/python/tensorrt-*-cp${PY_VERSION}-*.whl && \
    rm -rf /tmp/tensorrt

COPY . /root/ros2_ws/src
RUN pip3 install -r src/requirements.txt 
RUN rosdep install --from-paths src --ignore-src -r -y
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build

# Source the ROS2 setup file
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc
RUN echo "LD_LIBRARY_PATH=\"/usr/local/cuda-12.8/lib64:/usr/src/tensorrt/lib:/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH\"" >> ~/.bashrc

#libvndla_compiler.so not found fix
RUN wget -O - https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-dla-compiler/nvidia-l4t-dla-compiler_36.4.1-20241119120551_arm64.deb | dpkg-deb --fsys-tarfile - | sudo tar xv --strip-components=5 --directory=/usr/lib/aarch64-linux-gnu/nvidia/ ./usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so
CMD ["bash"]
