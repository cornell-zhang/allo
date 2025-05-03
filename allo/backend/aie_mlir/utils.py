import re
import numpy as np

import aie.ir as aie_ir
import allo._mlir._mlir_libs._mlir as allo_ir
from typing import Dict, List, Tuple
from ..._mlir.dialects import func as allo_func_d
from ..utils import format_str, format_code

from ..._mlir.ir import (
    InsertionPoint,
    FlatSymbolRefAttr,
    StringAttr,
)

aie_ctype_map = {
    "bf16": "bfloat16",
    "f32": "float",
    "f64": "double",
    "i8": "int8_t",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "i128": "__int128_t",  # unverified
    "ui1": "bool",
    "ui8": "uint8_t",
    "ui16": "unsigned short",
    "ui32": "unsigned int",
    "ui64": "unsigned long",
}

def inject_external_kernels(module: allo_ir.ir.Module) -> Tuple[Dict[str,bool], Dict]:
    '''
    Inject external kernels for compute cores.

    Return:
        - use_external_kernels: func_name -> use external
        - injected_kernels: kernel name -> external code snippet
    '''
    use_external_kernels = {}
    injected_kernels:Dict = {}
    
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, allo_func_d.FuncOp) and (
                "sym_visibility" not in func.attributes
                or func.attributes["sym_visibility"].value != "private"
            ):
                if func.attributes["sym_name"].value != "top":
                    func_name: str = func.attributes["sym_name"].value
                    use_external_kernels[func_name] = False
                    continue # fixme: crash when using external kernels
                    for block in func.regions[0].blocks:
                        for op in block.operations:
                            kernel_code, kernel_header = "", ""
                            # vec add/mul
                            if op.operation.name in {"linalg.add", "linalg.mul"}:
                                op_name = op.operation.name.split(".")[1]
                                dtype = str(op.inputs[0].type.element_type)
                                ctype = aie_ctype_map[dtype]
                                kernel_name = f"{op_name}_{dtype}_vector"
                                use_external_kernels[func_name] = True
                                kernel_code += f"void {kernel_name}({ctype} *A_in, {ctype} *B_in, {ctype} *C_out)"
                                kernel_code += " {\n"
                                kernel_code += f"  eltwise_v{op_name}<{ctype}, {ctype}, {np.prod(op.inputs[0].type.shape)}>(A_in, B_in, C_out);\n"
                                kernel_code += "}\n\n"
                            # matmul
                            elif op.operation.name == "linalg.matmul":
                                M, K = allo_ir.MemRefType(op.inputs[0].type).shape
                                _, N = allo_ir.MemRefType(op.inputs[1].type).shape
                                dtype = str(op.inputs[0].type.element_type)
                                out_dtype = str(op.outputs[0].type.element_type)
                                if (dtype, out_dtype) not in [
                                    ("i8", "i8"), ("i16", "i16"), ("i16", "i32"),
                                    ("bf16", "bf16"), ("bf16", "f32"),
                                ]:
                                    continue
                                kernel_name = f"matmul_scalar_{dtype}_{out_dtype}"
                                use_external_kernels[func_name] = True
                                kernel_header += f"#define DIM_M {M}\n"
                                kernel_header += f"#define DIM_N {N}\n"
                                kernel_header += f"#define DIM_K {K}\n"
                                kernel_header += f"#define {dtype}_{out_dtype}_ONLY\n"
                            else:
                                continue  
                            
                            # Inject AIE kernel
                            func_type = allo_func_d.FunctionType.get(
                                [op.inputs[0].type, op.inputs[1].type, op.outputs[0].type], [],
                            )
                            # replace operation
                            allo_func_d.CallOp(
                                [], FlatSymbolRefAttr.get(kernel_name),
                                [op.inputs[0], op.inputs[1], op.outputs[0]],
                                ip=InsertionPoint(op),
                            )
                            op.erase()
                            # register external kernel
                            if kernel_name in injected_kernels:
                                continue
                            injected_kernels[kernel_name] = (kernel_code, kernel_header)
                            kernel = allo_func_d.FuncOp(
                                kernel_name, func_type, ip=InsertionPoint(func),
                            )
                            kernel.attributes["sym_visibility"] = StringAttr.get("private")
    return use_external_kernels, injected_kernels

def classify_aie_functions(
    module: allo_ir.ir.Module
) -> Tuple[allo_func_d.FuncOp, Dict[str, List[allo_func_d.FuncOp]], List[allo_func_d.FuncOp]]:
    '''
    Classify the functions in allo module as
        - top
        - compute core functions
        - external kernel functions
    '''
    # top function
    top_func:allo_func_d.FuncOp = None
    # core functions
    core_func_groups:Dict[str, List[allo_func_d.FuncOp]] = {}
    # external functions
    external_funcs:List[allo_func_d.FuncOp] = []
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, allo_func_d.FuncOp):
                if ("sym_visibility" in func.attributes
                    and func.attributes["sym_visibility"].value == "private"
                ):
                    external_funcs.append(func)
                else:
                    if func.attributes["sym_name"].value == "top":
                        top_func = func
                    else:
                        func_name_w_id = func.attributes["sym_name"].value
                        func_name = re.match(r"^(.*?)_\d", func_name_w_id).group(1)
                        if func_name not in core_func_groups:
                            core_func_groups[func_name] = []
                        core_func_groups[func_name].append(func)
    return top_func, core_func_groups, external_funcs

def get_element_type(dtype_str: str) -> aie_ir.Type:
        if dtype_str == "i32":
            return aie_ir.IntegerType.get_signless(32)
        elif dtype_str == "i16":
            return aie_ir.IntegerType.get_signless(16)
        elif dtype_str == "i8":
            return aie_ir.IntegerType.get_signless(8)
        elif dtype_str == "f32":
            return aie_ir.F32Type.get()
        elif dtype_str == "f16":
            return aie_ir.F16Type.get()
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
 
def codegen_external_kernels(injected_kernels:Dict) -> str:
    code = """
// External kernels generated by Allo
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include "add.cc"
#include "mul.cc"
#include "mm.cc"
    """
    kernel_code = ""
    for kernel_snippet in injected_kernels.values():
        code += kernel_snippet[1]
        kernel_code += kernel_snippet[0]

    code += '\nextern "C" {\n\n'
    code += kernel_code
    code += '} // extern "C"\n'
    return code

np_supported_types = {
    "bf16": np.float32, # numpy does not support bf16
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

# ==================================================================================================
host_header = """
//=============================================================================
// Auto generated by Allo
//=============================================================================
#include "cxxopts.hpp"
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

int main(int argc, const char *argv[]) {
  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Vector Scalar Add Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << std::endl;
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Load the kernel
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << std::endl;
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
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>() << std::endl;
  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context." << std::endl;
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << std::endl;
  auto kernel = xrt::kernel(context, kernelName);

  // instruction
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  // output
  std::ofstream ofile("output.data");
  if (!ofile.is_open()) {
      std::cerr << "Error: Could not open output file.\\n";
      return 1;
  }

  // kernel arguments
  unsigned int opcode = 3;
"""

file_close_str = """  ofile.close();
  if (verbosity >= 1)
    std::cout << "Array has been written to output.data.\\n";
  return 0;
}
"""

def codegen_host(inputs, outputs):
    code = host_header
    with format_code(indent=2):
        # write input data
        for i, dtensor in enumerate(inputs):
            shape = dtensor.shape
            dtype = aie_ctype_map[str(dtensor.dtype)]
            code += format_str(f'std::ifstream ifile{i}("input{i}.data");')
            code += format_str(f"if (!ifile{i}.is_open()) {{")
            code += format_str(
                '  std::cerr << "Error: Could not open input file.\\n";', strip=False
            )
            code += format_str("  return 1;", strip=False)
            code += format_str("}")
            size = np.prod(shape)
            code += format_str(
                f"auto bo_in{i} = xrt::bo(device, {size} * sizeof({dtype}),"
            )
            with format_code(indent=24):
                code += format_str(
                    f"XRT_BO_FLAGS_HOST_ONLY, kernel.group_id({i + 3}));"
                )
            code += format_str(f"{dtype} *bufIn{i} = bo_in{i}.map<{dtype} *>();")
            code += format_str(f"std::vector<{dtype}> srcVec{i};")
            code += format_str(f"for (int i = 0; i < {size}; i++) {{")
            with format_code(indent=4):
                code += format_str(f"{dtype} num;")
                code += format_str(f"ifile{i} >> num;")
                code += format_str(f"srcVec{i}.push_back(num);")
            code += format_str("}")
            code += format_str(
                f"memcpy(bufIn{i}, srcVec{i}.data(), (srcVec{i}.size() * sizeof({dtype})));"
            )
        for i, dtensor in enumerate(outputs):
            shape = dtensor.shape
            dtype = aie_ctype_map[str(dtensor.dtype)]
            out_size = np.prod(shape)
            code += format_str(
                f"\nauto bo_out{i} = xrt::bo(device, {out_size} * sizeof({dtype}),",
                strip=False,
            )
            with format_code(indent=24):
                code += format_str(
                    f"XRT_BO_FLAGS_HOST_ONLY, kernel.group_id({len(inputs) + 2 + i}));"
                )
        code += format_str("if (verbosity >= 1)")
        code += format_str(
            '  std::cout << "Writing data into buffer objects.\\n";', strip=False
        )
        code += format_str("\nbo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);", strip=False)
        for i in range(len(inputs)):
            code += format_str(f"bo_in{i}.sync(XCL_BO_SYNC_BO_TO_DEVICE);")
        # run kernels
        code += format_str("if (verbosity >= 1)")
        code += format_str('  std::cout << "Running Kernel.\\n";', strip=False)
        code += format_str(
            "\nauto start = std::chrono::high_resolution_clock::now();", strip=False
        )
        inbufs = ", ".join([f"bo_in{i}" for i in range(len(inputs))])
        outbufs = ", ".join([f"bo_out{i}" for i in range(len(outputs))])
        code += format_str("// gid: (opcode, instr, instr_size, ...)")
        code += format_str(
            f"auto run = kernel(opcode, bo_instr, instr_v.size(), {inbufs}, {outbufs});"
        )
        code += format_str("run.wait();")
        code += format_str(
            "\nauto end = std::chrono::high_resolution_clock::now();", strip=False
        )
        code += format_str(
            "float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();"
        )
        code += format_str(
            'std::cout << "NPU execution time: " << npu_time << "us\\n";'
        )
        # get results
        for i, dtensor in enumerate(outputs):
            shape = dtensor.shape
            dtype = aie_ctype_map[str(dtensor.dtype)]
            out_size = np.prod(shape)
            code += format_str(
                f"\nbo_out{i}.sync(XCL_BO_SYNC_BO_FROM_DEVICE);", strip=False
            )
            code += format_str(f"{dtype} *bufOut{i} = bo_out{i}.map<{dtype} *>();")
            code += format_str(f"for (uint32_t i = 0; i < {out_size}; i++) {{")
            code += format_str(f'  ofile << *(bufOut{i} + i) << "\\n";', strip=False)
            code += format_str("}")
        code += format_str("\n// Close files", strip=False)
        for i in range(len(inputs)):
            code += format_str(f"ifile{i}.close();")
        code += file_close_str
    return code

def dfs_print(op, indent=0):
    op_name = str(op.name)
    if '.' in op_name:
        dialect = op_name.split('.')[0]
    else:
        dialect = "(x)"
    
    print('  ' * indent + f"Operation: {op_name},\tDialect: {dialect}")

    for region in op.regions:
        for block in region.blocks:
            for child_op in block.operations:
                dfs_print(child_op, indent + 1)

def print_module(module):
    dfs_print(module.operation)