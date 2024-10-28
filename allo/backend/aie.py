# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# mlir-aie commit: 8329b6

import os
import json
import textwrap
import numpy as np
import subprocess


def build_aie(mod, project):
    assert "MLIR_AIE_INSTALL_DIR" in os.environ, "Please set MLIR_AIE_INSTALL_DIR"
    assert "PEANO_INSTALL_DIR" in os.environ, "Please set PEANO_INSTALL_DIR"
    assert "LLVM_BUILD_DIR" in os.environ, "Please set LLVM_BUILD_DIR"
    # PATH=${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_BUILD_DIR}/bin:${PATH}
    code = """
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in1(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@in0] -> [@in1]([] [])
    aie.objectfifo @out0(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out1(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@out1] -> [@out0]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        %2 = aie.objectfifo.acquire @out1(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        %c0_0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c32 step %c1_1 {
          %4 = memref.load %1[%arg1] : memref<32xi32>
          %c1_i32 = arith.constant 1 : i32
          %5 = arith.addi %4, %c1_i32 : i32
          memref.store %5, %3[%arg1] : memref<32xi32>
        }
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out1(Produce, 1)
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in0} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 0 : i64, metadata = @out0} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @in0}
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}"""
    os.makedirs(os.path.join(project, "build"), exist_ok=True)
    with open(os.path.join(project, "top.mlir"), "w") as f:
        f.write(code)
    cmd = f"cd {project} && PYTHONPATH=$MLIR_AIE_INSTALL_DIR/python aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=build/final.xclbin --npu-insts-name=insts.txt top.mlir"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    path = os.path.dirname(__file__)
    path = os.path.join(path, "../harness/aie")
    os.system(f"cp -r {path}/* {project}")
    cmd = f"cd {project}/build && cmake .. -DTARGET_NAME=top && cmake --build . --config Release"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    cmd = f"cd {project}/build && ./top -x final.xclbin -i ../insts.txt -k MLIR_AIE"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()