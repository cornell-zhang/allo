# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# mlir-aie commit: 8329b6

import os
import json
import textwrap
import numpy as np
import subprocess

from .utils import format_str, format_code


def codegen_aie_mlir(mod):
    code = format_str("module {", indent=0)
    code += format_str("aie.device(npu1_1col) {", indent=2)
    # create tiles
    code += format_str("%tile_shim = aie.tile(0, 0)")
    code += format_str("%tile_mem = aie.tile(0, 1)")
    code += format_str("%tile_comp = aie.tile(0, 2)")
    # create object fifos
    # 2 means double buffer
    code += format_str("aie.objectfifo @in0(%tile_shim, {%tile_mem}, 2 : i32) : !aie.objectfifo<memref<64xi32>>")
    code += format_str("aie.objectfifo @in1(%tile_mem, {%tile_comp}, 2 : i32) : !aie.objectfifo<memref<64xi32>>")
    code += format_str("aie.objectfifo @out0(%tile_mem, {%tile_shim}, 2 : i32) : !aie.objectfifo<memref<64xi32>>")
    code += format_str("aie.objectfifo @out1(%tile_comp, {%tile_mem}, 2 : i32) : !aie.objectfifo<memref<64xi32>>")
    # construct connection
    code += format_str("aie.objectfifo.link [@in0] -> [@in1]([] [])")
    code += format_str("aie.objectfifo.link [@out1] -> [@out0]([] [])")
    # create core computation
    code += format_str("%core_0_2 = aie.core(%tile_comp) {")
    with format_code(indent=6):
        code += format_str("%c0 = arith.constant 0 : index")
        code += format_str("%c1 = arith.constant 1 : index")
        code += format_str("%c9223372036854775807 = arith.constant 9223372036854775807 : index")
        code += format_str("scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {")
        with format_code(indent=8):
            code += format_str("%fifo0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>")
            code += format_str("%local0 = aie.objectfifo.subview.access %fifo0[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>")
            code += format_str("%fifo1 = aie.objectfifo.acquire @out1(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>")
            code += format_str("%local1 = aie.objectfifo.subview.access %fifo1[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>")
            code += format_str("%c0_0 = arith.constant 0 : index")
            code += format_str("%c32 = arith.constant 64 : index")
            code += format_str("%c1_1 = arith.constant 1 : index")
            # code += format_str("scf.for %arg1 = %c0_0 to %c32 step %c1_1 {")
            # with format_code(indent=10):
            #     code += format_str("  %4 = memref.load %1[%arg1] : memref<64xi32>")
            #     code += format_str("  %c1_i32 = arith.constant 1 : i32")
            #     code += format_str("  %5 = arith.addi %4, %c1_i32 : i32")
            #     code += format_str("  memref.store %5, %3[%arg1] : memref<64xi32>")
            # code += format_str("}")
            mod_str = str(mod).replace("%arg0", "%local0")
            mod_str = mod_str.replace("%arg1", "%local1")
            with format_code(indent=4):
                for line in mod_str.splitlines()[2:-3]:
                    code += format_str(line, strip=False)
            code += format_str("aie.objectfifo.release @in1(Consume, 1)")
            code += format_str("aie.objectfifo.release @out1(Produce, 1)")
        code += format_str("}")
        code += format_str("aie.end")
    code += format_str("}")
    code += format_str("aiex.runtime_sequence(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {")
    with format_code(indent=6):
        code += format_str("aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in0} : memref<128xi32>")
        code += format_str("aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 0 : i64, metadata = @out0} : memref<128xi32>")
        code += format_str("aiex.npu.dma_wait {symbol = @in0}")
        code += format_str("aiex.npu.dma_wait {symbol = @out0}")
    code += format_str("}")
    code += format_str("}", indent=2)
    code += "}"
    return code


def build_aie(mod, project):
    assert "MLIR_AIE_INSTALL_DIR" in os.environ, "Please set MLIR_AIE_INSTALL_DIR"
    assert "PEANO_INSTALL_DIR" in os.environ, "Please set PEANO_INSTALL_DIR"
    assert "LLVM_BUILD_DIR" in os.environ, "Please set LLVM_BUILD_DIR"
    # PATH=${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_BUILD_DIR}/bin:${PATH}
    code = codegen_aie_mlir(mod)
    os.makedirs(os.path.join(project, "build"), exist_ok=True)
    with open(os.path.join(project, "top.mlir"), "w") as f:
        f.write(code)
    # sys.exit()
    # build mlir-aie
    cmd = f"cd {project} && PYTHONPATH=$MLIR_AIE_INSTALL_DIR/python aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=build/final.xclbin --npu-insts-name=insts.txt top.mlir"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to compile the MLIR-AIE code")
    path = os.path.dirname(__file__)
    path = os.path.join(path, "../harness/aie")
    os.system(f"cp -r {path}/* {project}")
    cmd = f"cd {project}/build && cmake .. -DTARGET_NAME=top && cmake --build . --config Release"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to build AIE project.")
    cmd = f"cd {project}/build && ./top -x final.xclbin -i ../insts.txt -k MLIR_AIE"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()