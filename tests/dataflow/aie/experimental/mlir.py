import subprocess
import os
import numpy as np

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


def read_tensor_from_file(dtype, shape, file_path):
    arr = np.fromfile(file_path, sep="\n", dtype=np_supported_types[str(dtype)])
    return arr.reshape(shape)


def call_mlir(project: str, output_dtype, trace_size: int, *args):
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
    for i, arg in enumerate(args[:-1]):
        with open(os.path.join(project, f"input{i}.data"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(i) for i in arg.flatten()]))
    cmd = f"cd {project} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE -p true --warmup 200 --test_iter 1000"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to execute AIE code.")
    result = read_tensor_from_file(
        output_dtype,
        args[-1].shape,
        f"{project}/output.data",
    )
    args[-1][:] = result


# fixme: update parameters as you need
from allo.ir.types import int16

TyI, TyO = int16, int16
M, N, K = 128, 128, 1024
A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
C = np.zeros((M, N)).astype(np.int16)

call_mlir("top.prj", TyI, 0, A, B, C)
np.testing.assert_allclose(C, A @ B, atol=1e-5)
print("PASSED!")
