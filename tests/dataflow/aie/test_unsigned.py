import os
from typing import Annotated

import numpy as np

import allo.dataflow as df
from allo.ir.types import uint8, int32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# from utils import analyze_trace, TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 1024


# Reference code starts
def reference_add_offset_uint8(
    x: Annotated[np.ndarray, "shape: (1024,)"],
    offset: Annotated[np.ndarray, "shape: (1,)"],
) -> Annotated[np.ndarray, "shape: (1024,)"]:
    return (x + int(offset[0])).astype(np.uint8)


# Reference code ends


def _test_add_offset_uint8():
    add_kernel = ExternalModule(
        top="add_offset_uint8",
        impl_path="canonical_scalar_allo.cc",
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = uint8
    Ty_scalar = int32
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly, Off: Ty_scalar[1]):
            add_kernel(A, B, Off)

    input_tensor = np.random.randint(
        0, 250, (1024,), dtype=np.uint8
    )  # Leave room for offset
    offset = np.array([2], dtype=np.int32)  # Small fixed offset to avoid overflow

    ref_output = reference_add_offset_uint8(input_tensor, offset)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie",
            profile=True,
            warmup=2000,
            num_iters=10000,
            trace=[("core", (0,))],
            trace_size=65536,
            # project=TOP_PRJ_ABS_DIR
        )
        output_allo = np.zeros((tensor_size,), dtype=np.uint8)
        mod(input_tensor, output_allo, offset)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_add_offset_uint8()
