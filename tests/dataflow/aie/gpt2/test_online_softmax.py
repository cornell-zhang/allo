import os
import allo
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

KERNEL_LIB_PATH = "/home/sf668/workspace/allo/tests/dataflow/aie/gpt2/kernels/"
np.random.seed(42)


def test_online_softmax(SEQ_LEN, HEAD_DIM, q_chunk_size=32, kv_chunk_size=32):
    init_softmax = ExternalModule(
        top="init_softmax",
        impl_path=KERNEL_LIB_PATH + "lut_softmax.cc",
        input_idx=[],
        output_idx=[0, 1],
    )

    online_softmax = ExternalModule(
        top="online_softmax",
        impl_path=KERNEL_LIB_PATH + "lut_softmax.cc",
        input_idx=[0, 1, 2],
        output_idx=[3, 4, 5, 6],
    )
