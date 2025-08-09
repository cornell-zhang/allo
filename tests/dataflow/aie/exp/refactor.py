import os
import re
import shutil


def modify_cpp_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    backup_path = os.path.join(os.path.dirname(filepath), "tmp.cpp")
    with open(backup_path, "w", encoding="utf-8") as bf:
        bf.write(content)
    print(f"Backed up original file to {backup_path}")

    pattern = re.compile(
        r"float\s+total_npu_time\s*=\s*0;\s*"
        r"float\s+npu_time_min\s*=\s*9999999;\s*"
        r"for\s*\(size_t\s+i\s*=\s*0;.*?\{\s*"
        r"auto\s+start\s*=.*?;\s*"
        r"(auto\s+run\s*=.*?;)\s*"
        r"run\.wait\(\);\s*"
        r"auto\s+end\s*=.*?;\s*"
        r"float\s+npu_time\s*=.*?;\s*"
        r"total_npu_time\s*\+=\s*npu_time;\s*"
        r"npu_time_min\s*=\s*\(npu_time\s*<\s*npu_time_min\)\s*\?\s*npu_time\s*:\s*npu_time_min;\s*"
        r"\}\s*"
        r'std::cout\s*<<\s*"Avg NPU execution time:.*?;\s*'
        r'std::cout\s*<<\s*"Min NPU execution time:.*?;',
        re.S,
    )

    def replacer(m):
        kernel_line = m.group(1)
        return (
            "float total_npu_time = 0;\n"
            "float npu_time_min = 9999999;\n"
            "float npu_time_max = 0;\n"
            "for (size_t i = 0; i < n_test_iterations; i++) {\n"
            "    auto start = std::chrono::high_resolution_clock::now();\n"
            f"    {kernel_line}\n"
            "    run.wait();\n"
            "    auto end = std::chrono::high_resolution_clock::now();\n"
            "    float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();\n"
            '    std::cout << "matmul time = " << npu_time << "us\\n";\n'
            "    total_npu_time += npu_time;\n"
            "    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;\n"
            "    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;\n"
            "}\n"
            'std::cout << "Avg NPU execution time: " << total_npu_time / n_test_iterations << "us\\n";\n'
            'std::cout << "Min NPU execution time: " << npu_time_min << "us\\n";\n'
            'std::cout << "Max NPU execution time: " << npu_time_max << "us\\n";'
        )

    content, n = pattern.subn(replacer, content)
    if n > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Modified {filepath}")
    else:
        print(f"No matching code found in {filepath}")


def modify_all_cpp(root_dir):
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file == "test.cpp":
                modify_cpp_file(os.path.join(dirpath, file))


if __name__ == "__main__":
    root_directory = "."
    modify_all_cpp(root_directory)
