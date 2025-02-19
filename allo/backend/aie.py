# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# mlir-aie commit: 8329b6
# pylint: disable=consider-using-with, bad-builtin, no-name-in-module, too-many-branches

import os
import subprocess
import re
import numpy as np
from .._mlir.ir import (
    IntegerAttr,
    IntegerType,
    DenseI64ArrayAttr,
    Context,
    RankedTensorType,
    FunctionType,
    TypeAttr,
    Location,
)
from .._mlir.dialects import func as func_d
from .._mlir.passmanager import PassManager as mlir_pass_manager

from .vitis import read_tensor_from_file
from ..utils import (
    get_func_inputs_outputs,
    get_dtype_and_shape_from_type,
    get_element_type_from_str,
)
from .utils import format_str, format_code
from .vitis import ctype_map


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
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  std::ofstream ofile("output.data");
  if (!ofile.is_open()) {
      std::cerr << "Error: Could not open output file.\\n";
      return 1;
  }

"""

file_close_str = """  ofile.close();
  if (verbosity >= 1)
    std::cout << "Array has been written to output.data.\\n";
  return 0;
}
"""


def codegen_host(input_args):
    code = host_header
    with format_code(indent=2):
        # write input data
        for i, (dtype, shape) in enumerate(input_args[:-1]):
            dtype = ctype_map[dtype]
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
        out_dtype, out_shape = input_args[-1]
        out_dtype = ctype_map[out_dtype]
        out_size = np.prod(out_shape)
        code += format_str(
            f"\nauto bo_out = xrt::bo(device, {out_size} * sizeof({out_dtype}),",
            strip=False,
        )
        with format_code(indent=24):
            code += format_str(
                f"XRT_BO_FLAGS_HOST_ONLY, kernel.group_id({len(input_args) + 2}));"
            )
        code += format_str("if (verbosity >= 1)")
        code += format_str(
            '  std::cout << "Writing data into buffer objects.\\n";', strip=False
        )
        code += format_str("\nbo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);", strip=False)
        for i in range(len(input_args) - 1):
            code += format_str(f"bo_in{i}.sync(XCL_BO_SYNC_BO_TO_DEVICE);")
        # run kernels
        code += format_str("if (verbosity >= 1)")
        code += format_str('  std::cout << "Running Kernel.\\n";', strip=False)
        code += format_str(
            "\nauto start = std::chrono::high_resolution_clock::now();", strip=False
        )
        code += format_str("unsigned int opcode = 3;", strip=False)
        inbufs = ", ".join([f"bo_in{i}" for i in range(len(input_args) - 1)])
        code += format_str("// gid: (opcode, instr, instr_size, ...)")
        code += format_str(
            f"auto run = kernel(opcode, bo_instr, instr_v.size(), {inbufs}, bo_out);"
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
        code += format_str("\nbo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);", strip=False)
        code += format_str(f"{out_dtype} *bufOut = bo_out.map<{out_dtype} *>();")
        code += format_str(f"for (uint32_t i = 0; i < {out_size}; i++) {{")
        code += format_str('  ofile << *(bufOut + i) << "\\n";', strip=False)
        code += format_str("}")
        code += format_str("\n// Close files", strip=False)
        for i in range(len(input_args) - 1):
            code += format_str(f"ifile{i}.close();")
        code += file_close_str
    return code


def codegen_aie_mlir(mod, orig_input_args, func_sizes, buf_dicts):
    input_args = orig_input_args.copy()
    code = format_str("module {", indent=0)
    mem_tile_size = 2 if len(input_args) > 2 else 1
    device = "npu1_2col" if len(input_args) > 2 else "npu1_1col"
    code += format_str(f"aie.device({device}) {{", indent=2)
    # create tiles
    code += format_str("%tile_shim = aie.tile(0, 0)")
    for mid in range(mem_tile_size):
        code += format_str(f"%tile_mem{mid} = aie.tile({mid}, 1)")
    # assert len(mapping) == 1, "Only support 1D mapping for now"
    # TODO: maybe use name of the function to support 2D?
    # number of function declaration except top
    funcs = list(mod.body.operations)[:-1]
    pe_size = len(funcs)
    buf_name_dicts = []
    for pid in range(pe_size):
        code += format_str(f"%tile_comp{pid} = aie.tile(0, {pid + 2})")
        buf_dict = buf_dicts[pid]
        buf_name_dict = {}
        for i, name in enumerate(buf_dict.keys()):
            tile_name = f"%tile_comp{pid}"
            new_name = f"{tile_name}_buf{i}"
            buf_name_dict[name] = new_name
            ele_type, shape = buf_dict[name]
            str_list = list(map(str, shape))
            str_list.append(ele_type)
            buf_type = f"memref<{'x'.join(map(str, str_list))}>"
            code += format_str(f"{new_name} = aie.buffer({tile_name}) : {buf_type}")
        buf_name_dicts.append(buf_name_dict)
    # update module and args
    for j, (ele_type, orig_shape) in enumerate(input_args):
        orig_ele_type = f"memref<{'x'.join(map(str, orig_shape))}x{ele_type}>"
        # TODO: need to deal with different sizes for different funcs
        shape = func_sizes[0][j]
        ele_type = f"memref<{'x'.join(map(str, shape))}x{ele_type}>"
        input_args[j] = (ele_type, orig_ele_type, shape, orig_shape)
    func_strs = list(map(str, funcs))
    # update buffers
    for pid in range(pe_size):
        func_str = func_strs[pid]
        buf_name_dict = buf_name_dicts[pid]
        # remove memref.alloc
        pattern_alloc = re.compile(r"^.*memref\.alloc.*\n?", re.MULTILINE)
        func_str = re.sub(pattern_alloc, "", func_str)
        # replace new buffer name
        pattern_boundary = r"(?<![\w.]){old}(?![\w.])"
        for name, new_name in buf_name_dict.items():
            escaped_name = re.escape(name)
            pattern = pattern_boundary.format(old=escaped_name)
            func_str = re.sub(pattern, new_name, func_str)
        func_strs[pid] = func_str
    # create object fifos
    # connect each argument to a separate mem tile
    linkings = [False] * len(input_args)
    for i, (in_type, orig_in_type, shape, orig_shape) in enumerate(input_args[:-1]):
        total_sizes = [0] * len(orig_shape)
        for sizes in func_sizes:
            for dim in range(len(orig_shape)):
                total_sizes[dim] += sizes[i][dim]
        for dim, orig_len in enumerate(orig_shape):
            if total_sizes[dim] <= orig_len:
                linkings[i] = True
                break
        if linkings[i]:
            # depth=2 means double buffer
            code += format_str(
                f"aie.objectfifo @in_sh{i}(%tile_shim, {{%tile_mem{i}}}, 2 : i32) : !aie.objectfifo<{orig_in_type}>"
            )
            for pid in range(pe_size):
                code += format_str(
                    f"aie.objectfifo @in{i}_p{pid}(%tile_mem{i}, {{%tile_comp{pid}}}, 2 : i32) : !aie.objectfifo<{in_type}>"
                )
            in_mem_str = ", ".join([f"@in{i}_p{pid}" for pid in range(pe_size)])
            shape_prod = np.prod(shape)
            in_mem_stride = list(range(0, shape_prod * pe_size, shape_prod))
            # (src_offsets, dst_offsets)
            code += format_str(
                f"aie.objectfifo.link [@in_sh{i}] -> [{in_mem_str}]([] {in_mem_stride})"
            )
        else:
            code += format_str(
                f"aie.objectfifo @in_sh{i}(%tile_shim, {{%tile_mem{i}}}, 2 : i32) : !aie.objectfifo<{orig_in_type}>"
            )
            in_tile_str = ", ".join([f"%tile_comp{pid}" for pid in range(pe_size)])
            code += format_str(
                f"aie.objectfifo @in{i}_p0(%tile_mem{i}, {{{in_tile_str}}}, 2 : i32) : !aie.objectfifo<{in_type}>"
            )
            code += format_str(f"aie.objectfifo.link [@in_sh{i}] -> [@in{i}_p0]([] [])")
    out_id = len(input_args) - 1
    out_type, orig_out_type, out_shape, orig_out_shape = input_args[-1]
    total_sizes = [0] * len(orig_out_shape)
    for sizes in func_sizes:
        for dim in range(len(orig_out_shape)):
            total_sizes[dim] += sizes[-1][dim]
    for dim, orig_out_len in enumerate(orig_out_shape):
        if total_sizes[dim] <= orig_out_len:
            linkings[-1] = True
            break
    if linkings[-1]:
        # output uses tile_mem0
        for pid in range(pe_size):
            code += format_str(
                f"aie.objectfifo @out_p{pid}(%tile_comp{pid}, {{%tile_mem0}}, 2 : i32) : !aie.objectfifo<{out_type}>"
            )
        code += format_str(
            f"aie.objectfifo @out_sh(%tile_mem0, {{%tile_shim}}, 2 : i32) : !aie.objectfifo<{orig_out_type}>"
        )
        out_mem_str = ", ".join([f"@out_p{pid}" for pid in range(pe_size)])
        shape_prod = np.prod(out_shape)
        out_mem_stride = list(range(0, shape_prod * pe_size, shape_prod))
        code += format_str(
            f"aie.objectfifo.link [{out_mem_str}] -> [@out_sh]({out_mem_stride} [])"
        )
    else:
        out_tile_str = ", ".join([f"%tile_comp{pid}" for pid in range(pe_size)])
        code += format_str(
            f"aie.objectfifo @out_sh(%tile_mem0, {{%tile_shim}}, 2 : i32) : !aie.objectfifo<{orig_out_type}>"
        )
        code += format_str(
            f"aie.objectfifo @out_p0({{{out_tile_str}}}, {{%tile_mem0}}, 2 : i32) : !aie.objectfifo<{out_type}>"
        )
        code += format_str("aie.objectfifo.link [@out_p0] -> [@out_sh]([] [])")
    # create core computation
    for pid, func_str in enumerate(func_strs):
        code += format_str(f"%core_0_{pid + 2} = aie.core(%tile_comp{pid}) {{")
        with format_code(indent=6):
            code += format_str("%c1000 = arith.constant 0 : index")
            code += format_str("%c1001 = arith.constant 1 : index")
            code += format_str(
                "%c9223372036854775807 = arith.constant 9223372036854775807 : index"
            )
            code += format_str(
                "scf.for %arg0 = %c1000 to %c9223372036854775807 step %c1001 {"
            )
            with format_code(indent=8):
                for i, (in_type, _, shape, _) in enumerate(input_args[:-1]):
                    code += format_str(
                        f"%fifo{i} = aie.objectfifo.acquire @in{i}_p{pid if linkings[i] else 0}(Consume, 1) : !aie.objectfifosubview<{in_type}>"
                    )
                    code += format_str(
                        f"%local{i} = aie.objectfifo.subview.access %fifo{i}[0] : !aie.objectfifosubview<{in_type}> -> {in_type}"
                    )
                    func_str = func_str.replace(f"%arg{i}", f"%local{i}")
                code += format_str(
                    f"%fifo_out = aie.objectfifo.acquire @out_p{pid}(Produce, 1) : !aie.objectfifosubview<{out_type}>"
                )
                code += format_str(
                    f"%local_out = aie.objectfifo.subview.access %fifo_out[0] : !aie.objectfifosubview<{out_type}> -> {out_type}"
                )
                func_str = func_str.replace(f"%arg{out_id}", "%local_out")
                with format_code(indent=4):
                    for line in func_str.splitlines()[1:-2]:
                        code += format_str(line, strip=False)
                for i in range(len(input_args[:-1])):
                    code += format_str(
                        f"aie.objectfifo.release @in{i}_p{pid if linkings[i] else 0}(Consume, 1)"
                    )
                code += format_str(f"aie.objectfifo.release @out_p{pid}(Produce, 1)")
            code += format_str("}")
            code += format_str("aie.end")
        code += format_str("}")
    in_args = ", ".join(
        [
            f"%arg{i}: {orig_in_type}"
            for i, (_, orig_in_type, _, _) in enumerate(input_args[:-1])
        ]
    )
    code += format_str(
        f"aiex.runtime_sequence({in_args}, %arg{out_id}: {orig_out_type}) {{"
    )
    with format_code(indent=6):
        for i, (_, orig_in_type, shape, _) in enumerate(input_args[:-1]):
            # (x, y, memref[offset][size][stride])
            # issue_token: MM2S-false, S2MM-true
            if len(shape) == 1:
                size_n_stride = f"[1, 1, 1, {shape[0] * (pe_size if linkings[i] else 1)}][0, 0, 0, 1]"
            else:
                size_n_stride = (
                    f"[1, 1, {shape[0] * pe_size}, {shape[1]}][0, 0, {shape[1]}, 1]"
                )
            code += format_str(
                f"aiex.npu.dma_memcpy_nd(0, 0, %arg{i}[0, 0, 0, 0]{size_n_stride}) {{id = {i + 1} : i64, issue_token = true, metadata = @in_sh{i}}} : {orig_in_type}"
            )
        if len(out_shape) == 1:
            out_size_n_stride = f"[1, 1, 1, {out_shape[0] * pe_size}][0, 0, 0, 1]"
        else:
            out_size_n_stride = f"[1, 1, {out_shape[0] * pe_size}, {out_shape[1]}][0, 0, {out_shape[1]}, 1]"
        code += format_str(
            f"aiex.npu.dma_memcpy_nd(0, 0, %arg{out_id}[0, 0, 0, 0]{out_size_n_stride}) {{id = 0 : i64, metadata = @out_sh}} : {orig_out_type}"
        )
        for i in range(len(input_args) - 1):
            code += format_str(f"aiex.npu.dma_wait {{symbol = @in_sh{i}}}")
        code += format_str("aiex.npu.dma_wait {symbol = @out_sh}")
    code += format_str("}")
    code += format_str("}", indent=2)
    code += "}"
    return code


def reindex_tensor_access(mod):
    ctx = mod.context
    funcs = list(mod.body.operations)[:-1]
    # func -> arg -> dim
    func_lower_bounds = []
    func_sizes = []
    for pi, func in enumerate(funcs):
        entry_block = func.regions[0].blocks[0]
        args = entry_block.arguments
        arg_types = args.types
        # TODO: might need some specialization for scalar input arg
        lower_bounds = [
            [float("inf") for _ in range(len(arg_type.shape))] for arg_type in arg_types
        ]
        sizes = [[0 for _ in range(len(arg_type.shape))] for arg_type in arg_types]
        for block in func.regions[0].blocks:
            for op in block.operations:
                if op.operation.name in {"tensor.extract_slice", "tensor.insert_slice"}:
                    operand_idx = (
                        0 if op.operation.name == "tensor.extract_slice" else 1
                    )
                    if op.operands[operand_idx] not in args:
                        continue
                    index = list(args).index(op.operands[operand_idx])
                    static_offsets = op.attributes["static_offsets"]
                    static_sizes = op.attributes["static_sizes"]
                    for i, (offset, size) in enumerate(
                        zip(static_offsets, static_sizes)
                    ):
                        lower_bounds[index][i] = min(lower_bounds[index][i], offset)
                        sizes[index][i] = max(sizes[index][i], size)
        for i, lower_bound in enumerate(lower_bounds):
            # Arguments never used with slice
            if lower_bound[0] == float("inf"):
                # If ever used, assume using entire tensor
                if len(list(args[i].uses)) > 0:
                    lower_bounds[i] = [0] * len(lower_bound)
                    sizes[i] = args[i].type.shape
                else:
                    lower_bounds[i] = [0] * len(lower_bound)
                    sizes[i] = [0] * len(lower_bound)

        func_lower_bounds.append(lower_bounds)
        func_sizes.append(sizes)

    for pi, func in enumerate(funcs):
        entry_block = func.regions[0].blocks[0]
        args = entry_block.arguments
        lower_bounds = func_lower_bounds[pi]
        sizes = func_sizes[pi]
        for block in func.regions[0].blocks:
            for op in block.operations:
                if op.operation.name == "tensor.extract_slice":
                    if op.operands[0] not in args:
                        continue
                    index = list(args).index(op.operands[0])
                    static_offsets = op.attributes["static_offsets"]
                    new_offsets = []
                    for i, offset in enumerate(static_offsets):
                        # TODO: need to support multi-dim mappings
                        # diff = pi * (op.operands[0].type.shape[0] // pe_size)
                        new_offset = offset - lower_bounds[index][i]
                        new_offset_attr = IntegerAttr.get(
                            IntegerType.get_signless(64, ctx), new_offset
                        )
                        new_offsets.append(new_offset_attr)
                    op.attributes["static_offsets"] = DenseI64ArrayAttr.get(
                        new_offsets, ctx
                    )
                elif op.operation.name == "tensor.insert_slice":
                    if op.operands[1] not in args:
                        continue
                    index = list(args).index(op.operands[1])
                    static_offsets = op.attributes["static_offsets"]
                    new_offsets = []
                    for i, offset in enumerate(static_offsets):
                        # TODO: need to support multi-dim mappings
                        # diff = pi * (op.operands[1].type.shape[0] // pe_size)
                        new_offset = offset - lower_bounds[index][i]
                        new_offset_attr = IntegerAttr.get(
                            IntegerType.get_signless(64, ctx), new_offset
                        )
                        new_offsets.append(new_offset_attr)
                    op.attributes["static_offsets"] = DenseI64ArrayAttr.get(
                        new_offsets, ctx
                    )
    return func_lower_bounds, func_sizes


def update_func_op_arg_types(
    func_op: func_d.FuncOp, input_args, new_shapes, context: Context
):
    old_func_type = func_op.function_type
    old_result_types = old_func_type.value.results
    new_input_types = []
    for (ele_type_str, _), shape in zip(input_args, new_shapes):
        elem_ty = get_element_type_from_str(ele_type_str, context)
        memref_ty = RankedTensorType.get(shape, elem_ty)
        new_input_types.append(memref_ty)
    new_func_type = FunctionType.get(new_input_types, old_result_types, context)
    new_type = TypeAttr.get(new_func_type, context)
    func_op.operation.attributes["function_type"] = new_type
    entry_block = func_op.entry_block
    for i, block_arg in enumerate(entry_block.arguments):
        block_arg.set_type(new_input_types[i])


def lower_tensor_to_memref(mod):
    passes = [
        # "linalg-generalize-named-ops",
        # "linalg-fuse-elementwise-ops",
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        "func.func(convert-linalg-to-affine-loops),lower-affine",
    ]
    pipeline = f'builtin.module({",".join(passes)})'
    with mod.context:
        mlir_pass_manager.parse(pipeline).run(mod.operation)


def record_local_buffer(mod):
    buf_dicts = []
    funcs = list(mod.body.operations)[:-1]
    for func in funcs:
        buf_dict = {}
        for block in func.regions[0].blocks:
            for op in block.operations:
                if op.operation.name == "memref.alloc":
                    name = op.result.get_name()
                    dtype, shape = get_dtype_and_shape_from_type(op.result.type)
                    buf_dict[name] = (dtype, shape)
        buf_dicts.append(buf_dict)
    return buf_dicts


class AIEModule:
    def __init__(self, module, top_func_name, project):
        self.module = module
        self.top_func_name = top_func_name
        # TODO: need to support multiple kernels
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp) and op.name.value != top_func_name:
                self.kernel_func = op
        self.project = project
        self.module = module

    def build(self):
        assert "MLIR_AIE_INSTALL_DIR" in os.environ, "Please set MLIR_AIE_INSTALL_DIR"
        assert "PEANO_INSTALL_DIR" in os.environ, "Please set PEANO_INSTALL_DIR"
        self.inputs, self.outputs = get_func_inputs_outputs(self.kernel_func)
        input_args = self.inputs + self.outputs
        _, func_sizes = reindex_tensor_access(self.module)
        with self.module.context as ctx, Location.unknown():
            for i, func_op in enumerate(list(self.module.body.operations)[:-1]):
                shapes = func_sizes[i]
                update_func_op_arg_types(func_op, input_args, shapes, ctx)
        lower_tensor_to_memref(self.module)
        buf_dicts = record_local_buffer(self.module)
        code = codegen_aie_mlir(self.module, input_args, func_sizes, buf_dicts)
        os.makedirs(os.path.join(self.project, "build"), exist_ok=True)
        with open(os.path.join(self.project, "top.mlir"), "w", encoding="utf-8") as f:
            f.write(code)
        # build mlir-aie
        cmd = f"cd {self.project} && PYTHONPATH=$MLIR_AIE_INSTALL_DIR/python aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=build/final.xclbin --npu-insts-name=insts.txt top.mlir"
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to compile the MLIR-AIE code")
        path = os.path.dirname(__file__)
        path = os.path.join(path, "../harness/aie")
        os.system(f"cp -r {path}/* {self.project}")
        host_code = codegen_host(input_args)
        with open(os.path.join(self.project, "test.cpp"), "w", encoding="utf-8") as f:
            f.write(host_code)
        cmd = f"cd {self.project}/build && cmake .. -DTARGET_NAME=top -DMLIR_AIE_DIR=$MLIR_AIE_INSTALL_DIR/.. && cmake --build . --config Release"
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to build AIE project.")
        return self

    def __call__(self, *args):
        # suppose the last argument is output
        for i, arg in enumerate(args[:-1]):
            with open(
                os.path.join(self.project, f"input{i}.data"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join([str(i) for i in arg.flatten()]))
        cmd = f"cd {self.project} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE"
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to execute AIE code.")
        result = read_tensor_from_file(
            self.inputs[-1][0], args[-1].shape, f"{self.project}/output.data"
        )
        args[-1][:] = result
