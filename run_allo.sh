#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper for conda run -n allo that sets the required LD_LIBRARY_PATH
# for libstdc++ compatibility on RHEL 8 (zhang-21).
#
# Usage:
#   ./run_allo.sh python tests/dataflow/catapult_synth_decoupled_2x1.py
#   ./run_allo.sh python -m pytest tests/dataflow/test_decoupled_mesh.py -v

CONDA_ENV_LIB="/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/lib"
MLIR_BUILD_LIB="/work/shared/users/phd/sk3463/projects/allo/mlir/build/tools/allo/_mlir"
export LD_LIBRARY_PATH="${MLIR_BUILD_LIB}:${CONDA_ENV_LIB}:${LD_LIBRARY_PATH}"

exec conda run -n allo "$@"
