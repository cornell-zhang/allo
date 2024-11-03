# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# mlir-aie commit: 8329b6

import os
import json
import textwrap
import numpy as np
import subprocess

from .utils import format_str, format_code


host_header = """
//=============================================================================
// Auto generated by Allo
//=============================================================================

#include <boost/program_options.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  po::options_description desc("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(desc);

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Load the kernel
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\\n";
  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\\n";
  auto kernel = xrt::kernel(context, kernelName);

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------
  constexpr int IN_SIZE = 128;
  constexpr int OUT_SIZE = 128;
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  std::ifstream inputFile("input0.data");
  std::ofstream outputFile("output.data");
  // Check if the file opened successfully
  if (!inputFile.is_open()) {
      std::cerr << "Error: Could not open input file.\\n";
      return 1;
  }
  if (!outputFile.is_open()) {
      std::cerr << "Error: Could not open output file.\\n";
      return 1;
  }
  auto bo_in0 = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\\n";

  uint32_t *bufIn0 = bo_in0.map<uint32_t *>();
  std::vector<uint32_t> srcVec0;
  for (int i = 0; i < IN_SIZE; i++) {
    int num;
    inputFile >> num;
    srcVec0.push_back(num);
  }
  memcpy(bufIn0, srcVec0.data(), (srcVec0.size() * sizeof(uint32_t)));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in0, bo_out);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *bufOut = bo_out.map<uint32_t *>();

  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    outputFile << *(bufOut + i) << "\\n";
  }

  // Close files
  inputFile.close();
  outputFile.close();
  if (verbosity >= 1)
    std::cout << "Array has been written to output.data.\\n";
  return 0;
}
"""

def codegen_host(mod):
    code = host_header
    return code


def codegen_aie_mlir(mod, input_args):
    code = format_str("module {", indent=0)
    code += format_str("aie.device(npu1_1col) {", indent=2)
    # create tiles
    code += format_str("%tile_shim = aie.tile(0, 0)")
    code += format_str("%tile_mem = aie.tile(0, 1)")
    code += format_str("%tile_comp = aie.tile(0, 2)")
    # create object fifos
    # 2 means double buffer
    in_type = input_args[0][1]
    out_type = input_args[1][1]
    code += format_str(f"aie.objectfifo @in0(%tile_shim, {{%tile_mem}}, 2 : i32) : !aie.objectfifo<{in_type}>")
    code += format_str(f"aie.objectfifo @in1(%tile_mem, {{%tile_comp}}, 2 : i32) : !aie.objectfifo<{in_type}>")
    code += format_str(f"aie.objectfifo @out0(%tile_mem, {{%tile_shim}}, 2 : i32) : !aie.objectfifo<{out_type}>")
    code += format_str(f"aie.objectfifo @out1(%tile_comp, {{%tile_mem}}, 2 : i32) : !aie.objectfifo<{out_type}>")
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
            code += format_str(f"%fifo0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<{in_type}>")
            code += format_str(f"%local0 = aie.objectfifo.subview.access %fifo0[0] : !aie.objectfifosubview<{in_type}> -> {in_type}")
            code += format_str(f"%fifo1 = aie.objectfifo.acquire @out1(Produce, 1) : !aie.objectfifosubview<{out_type}>")
            code += format_str(f"%local1 = aie.objectfifo.subview.access %fifo1[0] : !aie.objectfifosubview<{out_type}> -> {out_type}")
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
    code += format_str(f"aiex.runtime_sequence(%arg0: {in_type}, %arg1: {out_type}) {{")
    in_shape = input_args[0][2][0]
    out_shape = input_args[1][2][0]
    with format_code(indent=6):
        code += format_str(f"aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, {in_shape}][0, 0, 0, 1]) {{id = 1 : i64, issue_token = true, metadata = @in0}} : {in_type}")
        code += format_str(f"aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, {out_shape}][0, 0, 0, 1]) {{id = 0 : i64, metadata = @out0}} : {out_type}")
        code += format_str("aiex.npu.dma_wait {symbol = @in0}")
        code += format_str("aiex.npu.dma_wait {symbol = @out0}")
    code += format_str("}")
    code += format_str("}", indent=2)
    code += "}"
    return code


def build_aie(s, name, project):
    assert "MLIR_AIE_INSTALL_DIR" in os.environ, "Please set MLIR_AIE_INSTALL_DIR"
    assert "PEANO_INSTALL_DIR" in os.environ, "Please set PEANO_INSTALL_DIR"
    assert "LLVM_BUILD_DIR" in os.environ, "Please set LLVM_BUILD_DIR"
    # PATH=${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_BUILD_DIR}/bin:${PATH}
    mod = s.module
    input_args = []
    for idx, arg in enumerate(s.func_args[name]):
        dtype = s.top_func.attributes["function_type"].value.inputs[idx]
        shape = dtype.shape
        input_args.append((arg, dtype, shape))
    print(input_args)
    code = codegen_aie_mlir(mod, input_args)
    os.makedirs(os.path.join(project, "build"), exist_ok=True)
    with open(os.path.join(project, "top.mlir"), "w") as f:
        f.write(code)
    # build mlir-aie
    cmd = f"cd {project} && PYTHONPATH=$MLIR_AIE_INSTALL_DIR/python aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=build/final.xclbin --npu-insts-name=insts.txt top.mlir"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to compile the MLIR-AIE code")
    path = os.path.dirname(__file__)
    path = os.path.join(path, "../harness/aie")
    os.system(f"cp -r {path}/* {project}")
    host_code = codegen_host(mod)
    with open(os.path.join(project, "test.cpp"), "w") as f:
        f.write(host_code)
    cmd = f"cd {project}/build && cmake .. -DTARGET_NAME=top && cmake --build . --config Release"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to build AIE project.")
    return AIEModule(project, code)
    

class AIEModule:
    def __init__(
        self,
        project,
        code
    ):
        self.project = project
        self.code = code
        # func = find_func_in_module(self.module, self.top_func_name)
        # inputs, _ = get_func_inputs_outputs(func)

    def __call__(
        self, *args
    ):
        # suppose the last argument is output
        for i, arg in enumerate(args[:-1]):
            with open(os.path.join(self.project, f"input{i}.data"), "w") as f:
                f.write("\n".join([str(i) for i in arg.flatten()]))
        cmd = f"cd {self.project} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE"
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to execute AIE code.")
        with open(os.path.join(self.project, "output.data"), "r") as f:
            data = f.readlines()
        result = np.array([int(i) for i in data])
        args[-1][:] = result
        return