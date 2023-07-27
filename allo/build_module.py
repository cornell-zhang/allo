# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.passmanager import PassManager as mlir_pass_manager

from .context import get_context


def _mlir_lower_pipeline(module, **kwargs):
    hcl_d.loop_transformation(module)
    passes = ["affine-loop-normalize", "cse", "affine-simplify-structures"]
    if "canonicalize" in kwargs:
        passes += ["canonicalize"]
    if "lower_linalg" in kwargs:
        passes += ["convert-linalg-to-affine-loops"]
    pipeline = f'func.func({",".join(passes)})'
    try:
        with get_context():
            mlir_pass_manager.parse(pipeline).run(module)
        return module
    except Exception as e:
        print("Error: failed to run MLIR lower pipeline, printing module...")
        print(module)
        raise e
