# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install sudo

USER root

# install essentials
RUN sudo apt-get update && apt-get -y install git wget vim gdb gcc make software-properties-common libssl-dev

# install gcc-9
RUN sudo apt -y install build-essential && \
    sudo apt-get update && \
    sudo add-apt-repository ppa:ubuntu-toolchain-r/ppa -y && \
    sudo apt-get update && \
    sudo apt -y install gcc-9 g++-9 && \
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
                             --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    sudo update-alternatives --config gcc

# install cmake
WORKDIR /root/
RUN cd /root/ && wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9.tar.gz && \
    tar -xzvf cmake-3.27.9.tar.gz && \
    cd cmake-3.27.9 && \
    ./bootstrap && \
    make -j`nproc`
ENV PATH="${PATH}:/root/cmake-3.27.9/bin"

# install conda env
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p /root/miniconda && \
    eval "$(/root/miniconda/bin/conda shell.bash hook)" && \
    conda create --name hcl-dev python=3.12 -y && \
    conda activate hcl-dev

SHELL ["/root/miniconda/bin/conda", "run", "-n", "hcl-dev", "/bin/bash", "-c"]

# download llvm-project
RUN cd /root/ && git clone https://github.com/llvm/llvm-project.git

# install llvm
RUN cd /root/llvm-project && \
    git checkout tags/llvmorg-18-init && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy PyYAML dataclasses pybind11>=2.9.0  && \
    mkdir build && cd build && \
    cmake -G "Unix Makefiles" ../llvm -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86" \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=`which python3` && \
    make -j`nproc`

# download hcl-dialect
RUN cd /root/ && git clone https://github.com/cornell-zhang/hcl-dialect.git

# install hcl-dialect
RUN cd /root/hcl-dialect && \
    mkdir build && cd build && \
    cmake -G "Unix Makefiles" .. \
        -DMLIR_DIR=/root/llvm-project/build/lib/cmake/mlir \
        -DLLVM_EXTERNAL_LIT=/root/llvm-project/build/bin/llvm-lit \
        -DPYTHON_BINDING=ON \
        -DOPENSCOP=OFF \
        -DPython3_EXECUTABLE=`which python3` \
        -DCMAKE_CXX_FLAGS="-Wfatal-errors -std=c++17" && \
    make -j`nproc` && \
    cd tools/hcl/python_packages/hcl_core && \
    python3 -m pip install -e .

# initialize conda environment
ENV PATH="${PATH}:/root/miniconda/bin"
RUN conda init bash && \
    echo "conda activate hcl-dev" >> ~/.bashrc