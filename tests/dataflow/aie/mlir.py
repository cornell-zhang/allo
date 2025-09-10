import subprocess
import os
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16


def pack_int4(arr: np.ndarray) -> np.ndarray:
    arr = arr.flatten()
    arr_clipped = np.clip(arr, -8, 7).astype(np.int8)
    arr_u4 = (arr_clipped.astype(np.int8) & 0x0F).astype(np.uint8)
    if arr_u4.size % 2 != 0:
        arr_u4 = np.append(arr_u4, 0)
    low = arr_u4[0::2]
    high = arr_u4[1::2] << 4
    packed = (low | high).astype(np.uint8)
    return packed


np_supported_types = {
    "bf16": np.uint16,
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


def read_tensor_from_file(dtype, shape, file_path):
    arr = np.fromfile(file_path, dtype=np_supported_types[str(dtype)])
    if str(dtype) == "bf16":
        f32_arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return f32_arr.reshape(shape)
    return arr.reshape(shape)


def call_mlir(
    project: str,
    dtype_list: list,
    trace_size: int,
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
        if str(dtype_list[idx]) == "i4":
            arg = pack_int4(arg)
        with open(os.path.join(project, f"input{idx}.data"), "wb") as f:
            f.write(arg.tobytes())
    cmd = f"cd {project} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE -p false --warmup 200 --test_iter 1000"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to execute AIE code.")
    for idx in output_idx:
        result = read_tensor_from_file(
            dtype_list[idx],
            args[idx].shape,
            f"{project}/output{idx}.data",
        )
        args[idx][:] = result


if __name__ == "__main__":
    # fixme: update parameters as you need
    from allo.ir.types import int8, int16, int32, bfloat16

    M, N, K = 256, 256, 256
    A = (np.random.random((M, K)) * 0.1).astype(np_bfloat16)
    B = (np.random.random((K, N)) * 0.1).astype(np_bfloat16)
    C = np.zeros((M, N)).astype(np_bfloat16)

    call_mlir("top.prj", [bfloat16, bfloat16, bfloat16], 0, [0, 1], [2], A, B, C)
    np.testing.assert_allclose(
        C.astype(np.float32), (A @ B).astype(np.float32), atol=1e-2
    )
    print("PASSED!")
