<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Allo DSL

Please refer to [Documentation](https://chhzh123.github.io/allo-docs) for more details.


## Installation

Please clone the Allo repository to your local machine.

```bash
git clone https://github.com/cornell-zhang/allo.git
cd allo
```

We recommend creating a new conda environment for Allo. Since we are using the latest Python features, the minimum Python version is 3.12.

```bash
conda create -n allo python=3.12
conda activate allo
```


### Prerequisites

We need to first install the [LLVM project](https://github.com/llvm/llvm-project/tree/llvmorg-18-init) and the [hcl-mlir dialect](https://github.com/cornell-zhang/hcl-dialect). Users can choose to use our provided docker or build from source.

#### Docker

To simplify the installation process, we provide a docker image that has already installed the LLVM-18.x project.
Please pull the image from Docker Hub, **patch LLVM, and install the hcl dialect** as described above.

```bash
# * The LLVM is installed in /root/llvm-project in the docker image, which has already been patched
# * A prebuilt hcl-dialect is installed in /root/hcl-dialect, but please note that it is not up-to-date
#   You can pull the latest hcl-dialect using `git pull` and rebuild it if needed
docker pull chhzh123/hcl-dialect:llvm-18.x-py3.12
docker run --rm -it chhzh123/hcl-dialect:llvm-18.x-py3.12
```

#### Build from source

Users can also choose to build LLVM and the hcl dialect from source. Please follow the instructions below.

```bash
# Make sure you are under the correct Python environment
bash build.sh
```


### Install Allo

After installing LLVM and the hcl dialect, we can directly `pip install` Allo:

```bash
# Under the root directory of Allo
python3 -m pip install -e .
```
