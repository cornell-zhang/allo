#include "xcl2.hpp"
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_test_tpu;
    cl::CommandQueue q;

    // =========================================================================
    // 1. INITIALIZATION (Run Once)
    // =========================================================================
    std::cout << "Initializing OpenCL Runtime..." << std::endl;
    
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;

    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Programming device with: " << binaryFile << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin!" << std::endl;
            continue;
        }

        OCL_CHECK(err, krnl_test_tpu = cl::Kernel(program, "test_tpu", &err));
        valid_device = true;
        break;
    }

    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        return EXIT_FAILURE;
    }

    // =========================================================================
    // 2. INTERACTIVE LOOP
    // =========================================================================
    
    uint32_t op_in = 0;
    uint32_t addr_in = 0;
    int32_t val_in = 0;
    int32_t result_out = 0;

    // Create the Output Buffer (Arg 3) ONCE, reusable
    // We only need one buffer for the output pointer 'v3'
    cl::Buffer buffer_result(context, 
                             CL_MEM_WRITE_ONLY, 
                             sizeof(int32_t), 
                             nullptr, 
                             &err);

    while (true) {
        std::cout << "\n--- New Operation ---" << std::endl;
        std::cout << "Enter op  (uint8) [or 99 to exit]: ";
        if (!(std::cin >> op_in)) break; 
        if (op_in == 99) break;

        std::cout << "Enter addr  (uint8): ";
        std::cin >> addr_in;
        std::cout << "Enter val (int32): ";
        std::cin >> val_in;

        // --- Set Kernel Arguments (Matching kernel.cpp) ---
        int narg = 0;
        
        // Prepare scalar casts for sw_emu strictness
        uint8_t op_byte   = static_cast<uint8_t>(op_in);
        uint8_t addr_byte = static_cast<uint8_t>(addr_in);

        // Arg 0: v0 (uint8_t) -> Op Code
        OCL_CHECK(err, err = krnl_test_tpu.setArg(narg++, sizeof(uint8_t), &op_byte));

        // Arg 1: v1 (int32_t) -> Value (SCALAR, not buffer!)
        OCL_CHECK(err, err = krnl_test_tpu.setArg(narg++, sizeof(int32_t), &val_in));

        // Arg 2: v2 (uint8_t) -> Address Index
        OCL_CHECK(err, err = krnl_test_tpu.setArg(narg++, sizeof(uint8_t), &addr_byte));
        
        // Arg 3: v3 (int32_t*) -> Output Pointer (BUFFER)
        OCL_CHECK(err, err = krnl_test_tpu.setArg(narg++, buffer_result));

        // --- Execution ---
        
        // 1. Run Kernel
        // (Note: No input migration needed because args 0,1,2 are scalars)
        OCL_CHECK(err, err = q.enqueueTask(krnl_test_tpu));
        
        // 2. Read Result (Device -> Host)
        OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_result, CL_TRUE, 0, sizeof(int32_t), &result_out));
        
        // --- Output ---
        std::cout << "Kernel returned: " << result_out << std::endl;
    }

    return 0;
}
