# Local Installation

SLOAM was developed and tested using Ubuntu 20.04 with ROS Noetic and GCC 9.3. After all dependencies are installed you can continue following the main README file from `Build workspace`.

------

Create a folder called `thirdparty` to clone the packages that will be compiled with CMake.

```
mkdir ~/thirdparty
cd ~/thirdparty
```

### General dependencies

```
apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    gdb \
    libeigen3-dev \
    libglfw3-dev \
    libglew-dev \
    libtclap-dev \
    libtins-dev \
    libpcap-dev \
    libatlas-base-dev
```

### FMT
```
git clone https://github.com/fmtlib/fmt.git && \
    cd fmt && \
    git checkout 215f21a0382d325efa66df53fbfbfddb020a2234 && \
    mkdir fmt/build && cd fmt/build && \
    cmake .. && make && make install
```

### GLOG 
```
git clone https://github.com/google/glog.git && \
    cd glog && \ 
    git checkout ee6faf13b20de9536f456bd84584f4ab4db1ceb4 && \
    mkdir build && cd build && \
    cmake .. && make && make install
```

### Sophus
```
git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && \
    git checkout 49a7e1286910019f74fb4f0bb3e213c909f8e1b7 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && make && make install
```

### Ceres
```
git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout 206061a6ba02dc91286b18da48825f7a9ef561f0 && \
    mkdir build && cd build && cmake .. && make && make install
```

### ONNX
```
pip install pytest==6.2.1 onnx==1.10.1
cd /tmp && \
    git clone --recursive --branch v1.8.2 https://github.com/Microsoft/onnxruntime && \
    cd onnxruntime && \
    ./build.sh \
        --config RelWithDebInfo \
        --build_shared_lib \
        --build_wheel \
        --skip_tests \
        --parallel 3 && \
    cd build/Linux/RelWithDebInfo && \
    make install && \
    pip install dist/* && cd ..
```

### ROS Dependencies
```
apt-get install -y \
    ros-noetic-rviz \
    ros-noetic-tf2-eigen \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-image-geometry \
    ros-noetic-rqt-image-view \
    ros-noetic-eigen-conversions \
    ros-noetic-robot-localization \
    python3-catkin-tools \
    python3-osrf-pycommon
```