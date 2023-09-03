# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap

from ..ir.transform import find_func_in_module
from ..utils import get_func_inputs_outputs

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
    "u8": "unsigned char",
    "u16": "unsigned short",
    "u32": "unsigned int",
    "u64": "unsigned long",
}


def format_str(s, indent=4, strip=True):
    if strip:
        return textwrap.indent(textwrap.dedent(s).strip("\n"), " " * indent)
    return textwrap.indent(textwrap.dedent(s), " " * indent)


def codegen_host(top, module):
    func = find_func_in_module(module, top)
    inputs, outputs = get_func_inputs_outputs(func)
    # Get input/output types
    out_str = format_str(header, indent=0)
    out_str += format_str("\ncl::Kernel krnl_" + top + ";\n", strip=False)
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
    out_str += "\n"
    # Generate in/out buffers
    for i, (in_dtype, in_shape) in enumerate(inputs):
        in_dtype = ctype_map[in_dtype]
        in_shape = [str(i) for i in in_shape]
        out_str += format_str(
            f"size_t size_bytes_in{i} = sizeof({in_dtype}) * {' * '.join(in_shape)};\n",
            strip=False,
        )
        out_str += format_str(
            f"std::vector<{in_dtype}, aligned_allocator<{in_dtype}> > source_in{i}({' * '.join(in_shape)});\n",
            strip=False,
        )
    for i, (out_dtype, out_shape) in enumerate(outputs):
        out_dtype = ctype_map[out_dtype]
        out_shape = [str(i) for i in out_shape]
        out_str += format_str(
            f"size_t size_bytes_out{i} = sizeof({out_dtype}) * {' * '.join(out_shape)};\n",
            strip=False,
        )
        out_str += format_str(
            f"std::vector<{out_dtype}, aligned_allocator<{out_dtype}> > source_out{i}({' * '.join(out_shape)});\n",
            strip=False,
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
        f'\nOCL_CHECK(err, krnl_{top} = cl::Kernel(program, "{top}", &err));', 12, False
    )
    out_str += format_str(
        """
                valid_device = true;
                break; // we break because we found a valid device
            }
        }
        if (!valid_device) {
            std::cout << "Failed to program any device found, exit!\\n";
            exit(EXIT_FAILURE);
        }
        """,
        strip=False,
    )
    out_str += format_str(
        """
        // Allocate Buffer in Global Memory
        // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
        // Device-to-host communication
        """
    )
    out_str += "\n"
    for i in range(len(inputs)):
        out_str += format_str(
            f"OCL_CHECK(err, cl::Buffer buffer_in{i}(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in{i}, source_in{i}.data(), &err));\n",
            strip=False,
        )
    for i in range(len(outputs)):
        out_str += format_str(
            f"OCL_CHECK(err, cl::Buffer buffer_out{i}(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_bytes_out{i}, source_out{i}.data(), &err));\n",
            strip=False,
        )
    out_str += "\n"
    # Set kernel arguments
    for i in range(len(inputs)):
        out_str += format_str(
            f"OCL_CHECK(err, err = krnl_{top}.setArg({i}, buffer_in{i}));\n",
            strip=False,
        )
    for i in range(len(outputs)):
        out_str += format_str(
            f"OCL_CHECK(err, err = krnl_{top}.setArg({len(inputs) + i}, buffer_out{i}));\n",
            strip=False,
        )
    out_str += format_str("// Copy input data to device global memory\n", strip=False)
    buf_str = ", ".join([f"buffer_in{i}" for i in range(len(inputs))])
    out_str += format_str(
        "OCL_CHECK(err, err = q.enqueueMigrateMemObjects({"
        + buf_str
        + "}, 0 /* 0 means from host*/));\n",
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
    out_str += "\n\n"
    # Launch kernel
    out_str += format_str("// Launch the Kernel\n", strip=False)
    out_str += format_str(
        f"OCL_CHECK(err, err = q.enqueueTask(krnl_{top}, nullptr, &event));\n",
        strip=False,
    )
    out_str += "\n"
    out_str += format_str(
        "// Copy Result from Device Global Memory to Host Local Memory\n", strip=False
    )
    buf_str = ", ".join([f"buffer_out{i}" for i in range(len(outputs))])
    out_str += format_str(
        "OCL_CHECK(err, err = q.enqueueMigrateMemObjects({"
        + buf_str
        + "}, CL_MIGRATE_MEM_OBJECT_HOST));\n",
        strip=False,
    )
    out_str += format_str("q.finish();\n", strip=False)
    out_str += format_str("// OpenCL Host Code Ends\n", strip=False)
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
        f'std::cout << "| " << std::left << std::setw(24) << "{top}: "\n',
        strip=False,
    )
    out_str += format_str(
        '<< "|" << std::right << std::setw(24) << exe_time << " |\\n";\n',
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
        std::cout << "TEST PASSED\\n\\n";
        """,
    )
    out_str += "\n"
    out_str += format_str("return EXIT_SUCCESS;\n", strip=False)
    out_str += "}\n"
    return out_str


def postprocess_hls_code(hls_code):
    out_str = ""
    func_decl = False
    for line in hls_code.split("\n"):
        if line == "using namespace std;":
            out_str += line + "\n"
            # Add external function declaration
            out_str += '\nextern "C" {\n\n'
        elif line.startswith("void"):
            func_decl = True
            out_str += line + "\n"
        elif func_decl and line.startswith(") {"):
            func_decl = False
            out_str += line + "\n"
        elif func_decl:
            dtype, var = line.strip().split(" ")
            var = var.split("[")[0]
            out_str += "  " + dtype + " *" + var + ";\n"
        else:
            out_str += line + "\n"
    out_str += '} // extern "C"\n'
    return out_str
