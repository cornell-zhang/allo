# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Dockerfile for Allo with Xilinx Alveo FPGA support
# Base image: Official Xilinx runtime with XRT pre-installed

FROM xilinx/xilinx_runtime_base:alveo-2023.1-ubuntu-22.04
ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install essentials
RUN apt-get update && apt-get -y install \
    sudo git wget vim gdb gcc make \
    software-properties-common libssl-dev ninja-build \
    build-essential curl

# Install gcc-11 (default on Ubuntu 22.04, compatible with XRT)
RUN apt-get update && apt-get -y install gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Install CMake from pre-built binary (faster than building from source)
WORKDIR /root/
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.27.9-linux-x86_64.tar.gz && \
    rm cmake-3.27.9-linux-x86_64.tar.gz
ENV PATH="${PATH}:/root/cmake-3.27.9-linux-x86_64/bin"

# Install Miniforge (conda-forge based, no ToS restrictions)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /root/miniforge.sh && \
    bash /root/miniforge.sh -b -p /root/miniforge && \
    rm /root/miniforge.sh && \
    eval "$(/root/miniforge/bin/conda shell.bash hook)" && \
    conda create --name allo python=3.12 -y

# Set conda as default shell for subsequent RUN commands
SHELL ["/root/miniforge/bin/conda", "run", "-n", "allo", "/bin/bash", "-c"]

# Download LLVM/MLIR at specific commit
RUN cd /root/ && \
    git clone https://github.com/llvm/llvm-project.git && \
    cd llvm-project && \
    git checkout 6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb

# Build LLVM with MLIR and Python bindings
RUN cd /root/llvm-project && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy PyYAML dataclasses nanobind>=2.9 && \
    mkdir build && cd build && \
    cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="clang;mlir;openmp" \
        -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86" \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=`which python3` && \
    ninja

# Initialize conda environment
ENV PATH="${PATH}:/root/miniforge/bin"
RUN conda init bash && \
    echo "conda activate allo" >> ~/.bashrc

# Set LLVM environment variables
ENV LLVM_BUILD_DIR="/root/llvm-project/build"
ENV PATH="${PATH}:/root/llvm-project/build/bin"

# Install Allo Python dependencies
RUN python3 -m pip install \
    pybind11>=2.8.0 \
    xmltodict \
    tabulate \
    pytest \
    matplotlib \
    pandas \
    astpretty \
    packaging \
    psutil \
    sympy \
    rich \
    ml_dtypes

# Setup XRT environment on shell startup
RUN echo "source /opt/xilinx/xrt/setup.sh 2>/dev/null || true" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
