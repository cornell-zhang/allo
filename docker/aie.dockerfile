# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/AMDResearch/Ryzers/blob/b601f8789660f8418f7caeb3e8f3dbe9b17d9323/packages/npu/xdna/Dockerfile

FROM chhzh123/allo:latest

# Copy the scripts/ directory from the root of the allo project to /ryzers/scripts/ in the container
COPY scripts/ /ryzers/scripts/

ARG DRIVER_VERSION="0ad5aa3"
ARG PATCH_FILE="/ryzers/scripts//mlir-aie-patch.diff"

# required by rocm driver
RUN groupadd -f render && usermod -aG render root

WORKDIR /ryzers

# Suppress prompts in scripts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	git \
	curl \
	bash \
	wget

RUN git clone https://github.com/amd/xdna-driver && \
    cd xdna-driver && \
    git checkout $DRIVER_VERSION && \
    git submodule update --init --recursive

# Install build dependencies
RUN cd /ryzers/xdna-driver/tools && \
    apt-get update && \
    ./amdxdna_deps.sh -docker

# Build XRT
RUN cd /ryzers/xdna-driver/xrt/build && \
    export CXXFLAGS="-std=c++17" && \
    ./build.sh -npu -opt -noctest && \
    cd Release && \
    cmake --install .

# Build XDNA driver
RUN cd /ryzers/xdna-driver/build && \
    export CXXFLAGS="-std=c++17" && \
    ./build.sh -release && \
    ./build.sh -package

# Finally, save and install the XRT/XDNA debian packages
# these will also need to be installed on the host system
# if they haven't been already
RUN mkdir /ryzers/debs && \
    cp /ryzers/xdna-driver/build/Release/xrt_plugin.2.20.0_24.04-amd64-amdxdna.deb /ryzers/debs/ && \
    cp /ryzers/xdna-driver/xrt/build/Release/xrt_202520.2.20.0_24.04-amd64-base.deb /ryzers/debs/ && \
	dpkg -i /ryzers/debs/*.deb

RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && \
    conda activate allo && \
    python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0 && \
    python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl


RUN git clone https://github.com/Xilinx/mlir-aie.git "/ryzers/mlir-aie" && \
    cd "/ryzers/mlir-aie" && \
    git checkout 07320d6 && \
    patch -p1 < "$PATCH_FILE"
