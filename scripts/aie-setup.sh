#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

CURRENT_DIR="$(pwd)"

MLIR_AIE_CLONE_DIR="$(pwd)/.."
while [[ $# -gt 0 ]]; do
  case $1 in
    --clone-dir)
      MLIR_AIE_CLONE_DIR="$2"
      shift 2
      ;;
    *)
      echo "Warning: Unknown argument $1, ignoring"
      shift
      ;;
  esac
done

PATCH_FILE="$(pwd)/mlir-aie-patch.diff"
echo "$PATCH_FILE"

# === Clone mlir-aie and checkout specific commit ===
git clone https://github.com/Xilinx/mlir-aie.git "$MLIR_AIE_CLONE_DIR/mlir-aie"
cd "$MLIR_AIE_CLONE_DIR/mlir-aie"
git checkout 07320d6

# === Install Python requirements ===
python3 -m pip install -r python/requirements.txt
pre-commit install
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
python3 -m pip install -r python/requirements_ml.txt

# === Source environment ===
source utils/env_setup.sh

# === Apply patch to python/compiler/aiecc/main.py ===
patch -p1 < "$PATCH_FILE"

# === Copy patched main.py into install dir ===
INSTALL_PY_MAIN="$MLIR_AIE_INSTALL_DIR/python/aie/compiler/aiecc/main.py"
cp python/compiler/aiecc/main.py "$INSTALL_PY_MAIN"

# === Export MLIR_AIE_EXTERNAL_KERNEL_DIR ===
export MLIR_AIE_EXTERNAL_KERNEL_DIR="$MLIR_AIE_CLONE_DIR/mlir-aie/aie_kernels/"

# === Export RUNTIME_LIB_DIR ===
export RUNTIME_LIB_DIR="$MLIR_AIE_CLONE_DIR/mlir-aie/runtime_lib/"

cd "$CURRENT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

echo -e "${GREEN}=== AIE setup completed successfully ===${RESET}"

echo ""
echo -e "${YELLOW}>>> Please note: Each time you activate your environment, you need to export the following variables:${RESET}"
echo -e "${CYAN}export PATH=$MLIR_AIE_INSTALL_DIR/bin:\$PATH${RESET}"
echo -e "${CYAN}export MLIR_AIE_INSTALL_DIR=$MLIR_AIE_INSTALL_DIR${RESET}"
echo -e "${CYAN}export PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR${RESET}"
echo -e "${CYAN}export MLIR_AIE_EXTERNAL_KERNEL_DIR=$MLIR_AIE_EXTERNAL_KERNEL_DIR${RESET}"
echo -e "${CYAN}export RUNTIME_LIB_DIR=$RUNTIME_LIB_DIR${RESET}"
echo -e "${CYAN}export PYTHONPATH=$MLIR_AIE_INSTALL_DIR/python:\$PYTHONPATH${RESET}"
echo ""