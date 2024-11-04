# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np

from .utils import format_str
from ..ir.transform import find_func_in_module
from ..utils import get_func_inputs_outputs, get_clostest_pow2, np_supported_types

header = """
//=============================================================================
// Auto generated by Allo
//=============================================================================

// OpenCL utility layer include
#include "xcl2.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>
"""

main_header = """
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Program program;
"""

dtype_size_map = {
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
}

ctype_map = {
    "f32": "float",
    "f64": "double",
    "i8": "char",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "i128": "__int128_t",  # unverified
    "ui1": "bool",
    "ui8": "unsigned char",
    "ui16": "unsigned short",
    "ui32": "unsigned int",
    "ui64": "unsigned long",
}


# pylint: disable=too-many-branches
def codegen_host(top, module):
    # Reference: https://github.com/Xilinx/Vitis_Accel_Examples/blob/main/sys_opt/kernel_swap/src/host.cpp
    func = find_func_in_module(module, top)
    inputs, outputs = get_func_inputs_outputs(func)
    # Get input/output types
    out_str = format_str(header, indent=0, strip=False)
    for i in range(len(inputs)):
        out_str += format_str(f'#include "input_{i}.h"', indent=0, strip=False)
    out_str += format_str(main_header, indent=0, strip=False)
    out_str += format_str("cl::Kernel krnl_" + top + ";\n", strip=False)
    out_str += format_str(
        """
        // Allocate Memory in Host Memory
        // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
        // hood user ptr is used if it is properly aligned. when not aligned, runtime had no choice
        // but to create its own host side buffer. So it is recommended to use this allocator if
        // user wish to create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
        // boundary. It will ensure that user buffer is used when user create Buffer/Mem object with
        // CL_MEM_USE_HOST_PTR
        """
    )
    # Generate in/out buffers
    for i, (in_dtype, in_shape) in enumerate(inputs):
        if in_dtype in ctype_map:
            in_dtype = ctype_map[in_dtype]
        elif in_dtype.startswith("i") or in_dtype.startswith("ui"):
            prefix, bitwidth = in_dtype.split("i")
            if int(bitwidth) == 1:
                in_dtype = "bool"
            else:
                new_int_type = f"{prefix}i{max(get_clostest_pow2(int(bitwidth)), 8)}"
                in_dtype = ctype_map[new_int_type]
        elif in_dtype.startswith("fixed") or in_dtype.startswith("ufixed"):
            in_dtype = "float"
        else:
            raise ValueError(f"Unsupported input type: {in_dtype}")
        in_shape = [str(i) for i in in_shape]
        if len(in_shape) == 0:
            # scalar
            out_str += format_str(
                f"{in_dtype} source_in{i} = in_data_{i};", strip=False
            )
        else:
            out_str += format_str(
                f"size_t size_bytes_in{i} = sizeof({in_dtype}) * {' * '.join(in_shape)};",
                strip=False,
            )
            out_str += format_str(
                f"std::vector<{in_dtype}, aligned_allocator<{in_dtype}> > source_in{i}(in_data_{i}, in_data_{i} + {' * '.join(in_shape)});",
                strip=False,
            )
    for i, (out_dtype, out_shape) in enumerate(outputs):
        if out_dtype in ctype_map:
            out_dtype = ctype_map[out_dtype]
        elif out_dtype.startswith("i") or out_dtype.startswith("ui"):
            prefix, bitwidth = out_dtype.split("i")
            new_int_type = f"{prefix}i{max(get_clostest_pow2(int(bitwidth)), 8)}"
            out_dtype = ctype_map[new_int_type]
        elif out_dtype.startswith("fixed") or out_dtype.startswith("ufixed"):
            out_dtype = "float"
        else:
            raise ValueError(f"Unsupported input type: {out_dtype}")
        out_shape = [str(i) for i in out_shape]
        out_str += format_str(
            f"size_t size_bytes_out{i} = sizeof({out_dtype}) * {' * '.join(out_shape)};\n",
            strip=False,
        )
        out_str += format_str(
            f"std::vector<{out_dtype}, aligned_allocator<{out_dtype}> > source_out{i}({' * '.join(out_shape)});\n",
            strip=False,
        )
        out_str += format_str(
            f"std::fill(source_out{i}.begin(), source_out{i}.end(), 0);\n", strip=False
        )
    out_str += "\n"
    # Generate OpenCL host code
    out_str += format_str(
        """
        // OPENCL HOST CODE AREA START
        // get_xil_devices() is a utility API which will find the xilinx
        // platforms and will return list of devices connected to Xilinx platform
        auto devices = xcl::get_xil_devices();
        // read_binary_file() is a utility API which will load the binaryFile
        // and will return the pointer to file buffer.
        auto fileBuf = xcl::read_binary_file(binaryFile);
        cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
        bool valid_device = false;
        for (unsigned int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            // Creating Context and Command Queue for selected Device
            OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
            OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
            std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            cl::Program program(context, {device}, bins, nullptr, &err);
            if (err != CL_SUCCESS) {
                std::cout << "Failed to program device[" << i << "] with xclbin file!\\n";
            } else {
                std::cout << "Device[" << i << "]: program successful!\\n";
        """
    )
    out_str += format_str(
        f'OCL_CHECK(err, krnl_{top} = cl::Kernel(program, "{top}", &err));', 12, False
    )
    out_str += format_str(
        """            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\\n";
        exit(EXIT_FAILURE);
    }
    """,
        strip=False,
        indent=0,
    )
    out_str += format_str(
        """
        // Allocate Buffer in Global Memory
        // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
        // Device-to-host communication
        """
    )
    for i, (in_dtype, in_shape) in enumerate(inputs):
        if i == len(inputs) - 1 and len(outputs) == 0:
            # suppose the last input is also the output
            flag = "CL_MEM_READ_WRITE"
        else:
            flag = "CL_MEM_READ_ONLY"
        if len(in_shape) != 0:
            out_str += format_str(
                f"OCL_CHECK(err, cl::Buffer buffer_in{i}(context, CL_MEM_USE_HOST_PTR | {flag}, size_bytes_in{i}, source_in{i}.data(), &err));",
                strip=False,
            )
    for i in range(len(outputs)):
        out_str += format_str(
            f"OCL_CHECK(err, cl::Buffer buffer_out{i}(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_bytes_out{i}, source_out{i}.data(), &err));",
            strip=False,
        )
    out_str += "\n"
    # Set kernel arguments
    buf_str = ""
    for i, (in_dtype, in_shape) in enumerate(inputs):
        if len(in_shape) == 0:
            # scalar
            out_str += format_str(
                f"OCL_CHECK(err, err = krnl_{top}.setArg({i}, source_in{i}));",
                strip=False,
            )
        else:
            out_str += format_str(
                f"OCL_CHECK(err, err = krnl_{top}.setArg({i}, buffer_in{i}));",
                strip=False,
            )
            buf_str += f"buffer_in{i}, "
    for i in range(len(outputs)):
        out_str += format_str(
            f"OCL_CHECK(err, err = krnl_{top}.setArg({len(inputs) + i}, buffer_out{i}));",
            strip=False,
        )
    out_str += format_str("// Copy input data to device global memory", strip=False)
    buf_str = buf_str.strip(", ")
    out_str += format_str(
        "OCL_CHECK(err, err = q.enqueueMigrateMemObjects({"
        + buf_str
        + "}, 0 /* 0 means from host*/));",
        strip=False,
    )
    out_str += "\n"
    out_str += format_str(
        """
    cl::Event event;
    uint64_t nstimestart, nstimeend;
    std::cout << "|-------------------------+-------------------------|\\n"
              << "| Kernel                  |    Wall-Clock Time (ns) |\\n"
              << "|-------------------------+-------------------------|\\n";
    """
    )
    out_str += "\n"
    # Launch kernel
    out_str += format_str("// Launch the Kernel", strip=False)
    out_str += format_str(
        f"OCL_CHECK(err, err = q.enqueueTask(krnl_{top}, nullptr, &event));",
        strip=False,
    )
    out_str += "\n"
    out_str += format_str(
        "// Copy Result from Device Global Memory to Host Local Memory",
        strip=False,
    )
    if len(outputs) > 0:
        out_str += format_str(
            "OCL_CHECK(err, err = q.enqueueMigrateMemObjects({"
            + ", ".join([f"buffer_out{i}" for i in range(len(outputs))])
            + "}, CL_MIGRATE_MEM_OBJECT_HOST));",
            strip=False,
        )
    else:
        out_str += format_str(
            "OCL_CHECK(err, err = q.enqueueMigrateMemObjects({"
            + ", ".join([f"buffer_in{len(inputs) - 1}"])
            + "}, CL_MIGRATE_MEM_OBJECT_HOST));",
            strip=False,
        )
    out_str += format_str("q.finish();", strip=False)
    out_str += format_str("// OpenCL Host Code Ends", strip=False)
    out_str += "\n"
    # Timing
    out_str += format_str(
        """
        // Get the execution time
        OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        auto exe_time = nstimeend - nstimestart;
        """
    )
    out_str += "\n"
    out_str += format_str(
        f'std::cout << "| " << std::left << std::setw(24) << "{top} "',
        strip=False,
    )
    out_str += format_str(
        '<< "|" << std::right << std::setw(24) << exe_time << " |\\n";',
        strip=False,
        indent=14,
    )
    out_str += format_str(
        """
        std::cout << "|-------------------------+-------------------------|\\n";
        std::cout << "Note: Wall Clock Time is meaningful for real hardware execution "
                  << "only, not for emulation.\\n";
        std::cout << "Please refer to profile summary for kernel execution time for "
                  << "hardware emulation.\\n";
        std::cout << "Finished execution!\\n\\n";
        """,
    )
    out_str += "\n\n"
    assert len(outputs) <= 1, "Only support one output for now"
    if len(outputs) == 0:
        out_buf = "source_in" + str(len(inputs) - 1)
    else:
        out_buf = "source_out" + str(len(outputs) - 1)
    out_str += format_str(
        f"""    // Write the output data to file
    std::ofstream ofile;
    ofile.open("output.data");
    if (!ofile) {{
        std::cerr << "Failed to open output file!" << std::endl;
        return EXIT_FAILURE;
    }}
    for (unsigned i = 0; i < {out_buf}.size(); i++) {{
        ofile << {out_buf}[i] << std::endl;
    }}
    ofile.close();
    """,
        strip=False,
        indent=0,
    )
    out_str += format_str("return EXIT_SUCCESS;", strip=False)
    out_str += "}\n"
    return out_str


def postprocess_hls_code(hls_code, top=None):
    out_str = ""
    func_decl = False
    has_endif = False
    func_args = []
    for line in hls_code.split("\n"):
        if line == "using namespace std;" or line.startswith("#ifndef"):
            out_str += line + "\n"
            # Add external function declaration
            out_str += '\nextern "C" {\n\n'
        elif line.startswith(f"void {top}"):
            func_decl = True
            out_str += line + "\n"
        elif func_decl and line.startswith(") {"):
            func_decl = False
            out_str += line + "\n"
            # Add extra interfaces
            for i, arg in enumerate(func_args):
                out_str += f"  #pragma HLS interface m_axi port={arg} offset=slave bundle=gmem{i}\n"
        elif func_decl:
            dtype, var = line.strip().rsplit(" ", 1)
            comma = "," if var[-1] == "," else ""
            if "[" in var:  # array
                var = var.split("[")[0]
                out_str += "  " + dtype + " *" + var + f"{comma}\n"
                # only add array to interface
                func_args.append(var)
            else:  # scalar
                var = var.split(",")[0]
                out_str += "  " + dtype + " " + var + f"{comma}\n"
        elif line.startswith("#endif"):
            out_str += '} // extern "C"\n\n'
            out_str += line + "\n"
            has_endif = True
        else:
            out_str += line + "\n"
    if not has_endif:
        out_str += '} // extern "C"\n'
    return out_str


def generate_description_file(top, src_path, dst_path):
    with open(src_path, "r", encoding="utf-8") as f:
        desc = f.read()
    desc = desc.replace("top", top)
    desc = json.loads(desc)
    desc["containers"][0]["ldclflags"] += "  --kernel_frequency 300"
    with open(dst_path, "w", encoding="utf-8") as outfile:
        json.dump(desc, outfile, indent=4)


def update_makefile(file_name, ext_libs):
    with open(file_name, "r", encoding="utf-8") as f:
        makefile = f.read()
    cpp_files = ["kernel.cpp"]
    for lib in ext_libs:
        for impl_path in lib.impls:
            cpp_files.append(impl_path.split("/")[-1])
    makefile = makefile.replace("kernel.cpp", " ".join(cpp_files))
    with open(file_name, "w", encoding="utf-8") as outfile:
        outfile.write(makefile)


def write_tensor_to_file(tensor, dtype, shape, name, file_path):
    # generate C buffers
    with open(file_path, "w", encoding="utf-8") as f:
        if len(shape) == 0:
            # scalar
            f.write(f"const {ctype_map[dtype]} {name} = {tensor};\n")
        else:
            f.write(f"const {ctype_map[dtype]} {name}")
            # pylint: disable=bad-builtin
            f.write(f"[{', '.join(map(str, shape))}] = {{")
            f.write(", ".join([str(i) for i in tensor.flatten()]))
            f.write("};\n")


def read_tensor_from_file(dtype, shape, file_path):
    arr = np.fromfile(file_path, sep="\n", dtype=np_supported_types[dtype])
    return arr.reshape(shape)
