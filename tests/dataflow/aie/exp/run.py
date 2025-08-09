import subprocess
import os
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

np_supported_types = {
    "bf16": np.float32,  # numpy does not support bf16
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "ui1": np.bool_,
    "ui8": np.uint8,
    "ui16": np.uint16,
    "ui32": np.uint32,
    "ui64": np.uint64,
}


def call_mlir(
    project: str,
    dtype_list: list,
    input_idx: list[int],
    output_idx: list[int],
    *args,
):
    # generate insts.txt
    cmd = f"cd {project} && aiecc.py --alloc-scheme=basic-sequential --aie-generate-xclbin --no-compile-host --xclbin-name=build/final.xclbin --no-xchesscc --no-xbridge --peano ${{PEANO_INSTALL_DIR}} --aie-generate-npu-insts --npu-insts-name=insts.txt top.mlir"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to compile the MLIR-AIE code")
    cmd = f"cd {project}/build && cmake .. -DTARGET_NAME=top -DMLIR_AIE_DIR=$RUNTIME_LIB_DIR/.. && cmake --build . --config Release"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to build AIE project.")
    # suppose the last argument is output
    for idx in input_idx:
        arg = args[idx]
        with open(
            os.path.join(project, f"input{idx}.data"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join([str(i) for i in arg.flatten()]))
    cmd = f"cd {project} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE -p true --warmup 10 --test_iter 400"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to execute AIE code.")


# fixme: update parameters as you need
from allo.ir.types import bfloat16, int16, int8


def run(M, N, K, dtype):
    if dtype is bfloat16:
        A = np.random.random((M, K)).astype(np_bfloat16)
        B = np.random.random((K, N)).astype(np_bfloat16)
        C = np.zeros((M, N)).astype(np_bfloat16)
    elif dtype is int8:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int8)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
    elif dtype is int16:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
    else:
        raise ValueError(f"unsupported data type {dtype}")
    try:
        if dtype is bfloat16:
            call_mlir(
                f"gemm_{M}x{N}x{K}.prj",
                [dtype, dtype, dtype],
                [0, 1],
                [2],
                A,
                B,
                C,
            )
        else:
            call_mlir(
                f"gemm_{M}x{N}x{K}_{dtype}.prj",
                [dtype, dtype, dtype],
                [0, 1],
                [2],
                A,
                B,
                C,
            )
    except:
        print(f"M={M},N={N},K={K} exe failed")


K_list = [256, 512, 1024, 2048]
M_list = [256, 512, 1024, 2048]
N_list = [256, 512, 1024, 2048]
for M_ in M_list:
    for N_ in N_list:
        for K_ in K_list:
            run(M_, N_, K_, bfloat16)
# run(256, 256, 256, int16)
