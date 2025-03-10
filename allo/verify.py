# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import difflib

try:
    import past
except ImportError:
    pass


def verify(schedule_a, schedule_b):
    """
    Run PAST verifier on the two schedules, returning whether they are equivalent.

    If equivalence fails, output a diff of the generated code files to help diagnose the
    source of the mismatch.
    """
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)

    prog_a_path = os.path.join(temp_dir, "a.c")
    prog_b_path = os.path.join(temp_dir, "b.c")

    mod_a = schedule_a.build(target="vhls")
    mod_b = schedule_b.build(target="vhls")

    with open(prog_a_path, "w", encoding="utf-8") as f:
        f.write(str(mod_a))
    with open(prog_b_path, "w", encoding="utf-8") as f:
        f.write(str(mod_b))

    add_pocc_pragmas(prog_a_path)
    add_pocc_pragmas(prog_b_path)
    replace_unsupported_types(prog_a_path)
    replace_unsupported_types(prog_b_path)

    # detect output vars
    out_a = find_live_out_variable(prog_a_path, schedule_a.top_func_name)
    out_b = find_live_out_variable(prog_b_path, schedule_b.top_func_name)
    print(out_a, out_b)

    # if output vars have different names
    if out_a != out_b:
        rewrite_output_variable(prog_a_path, out_a, "live_out")
        rewrite_output_variable(prog_b_path, out_b, "live_out")
        live_out = "live_out"
    else:
        live_out = out_a

    is_equivalent = past.verify(prog_a_path, prog_b_path, live_out)

    # if not equivalent, produce a diff of the two programs
    if not is_equivalent:
        with open(prog_a_path, "r", encoding="utf-8") as f:
            code_a = f.readlines()
        with open(prog_b_path, "r", encoding="utf-8") as f:
            code_b = f.readlines()
        diff = difflib.unified_diff(
            code_a,
            code_b,
            fromfile="Program A (Schedule A)",
            tofile="Program B (Schedule B)",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        print("Verifier reported non-equivalence between schedules.")
        print("Differences between generated programs:")
        print(diff_text)
        print(f"Detected output variable in schedule A: {out_a}")
        print(f"Detected output variable in schedule B: {out_b}")

    return is_equivalent


def rewrite_output_variable(file_path, old_var, new_var):
    """
    Rewrite the file at file_path by replacing all whole-word occurrences of old_var with new_var,
    but (if possible) only inside the call block—the region between
    "#pragma pocc-region-start" and "#pragma pocc-region-end".
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "#pragma pocc-region-start" in content and "#pragma pocc-region-end" in content:
        # only rewrite the call region
        pattern = re.compile(
            r"(#pragma\s+pocc-region-start\s*(?:\{)?\s*)(.*?)(\s*(?:\})?\s*#pragma\s+pocc-region-end)",
            re.DOTALL,
        )

        def repl(match):
            start, block, end = match.groups()
            new_block = re.sub(r"\b" + re.escape(old_var) + r"\b", new_var, block)
            return start + new_block + end

        new_content = pattern.sub(repl, content)
    else:
        # global rewrite: not in call region
        new_content = re.sub(r"\b" + re.escape(old_var) + r"\b", new_var, content)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def find_live_out_variable(file_path, top_function_name):
    """
    Identify the live-out variable of a C kernel function.

    Args:
        file_path (str): Path to the C/C++ code file
        top_function_name (str): Name of the top-level function to analyze

    Returns:
        str: The name of the live-out variable
    """
    # Read the file
    with open(file_path, "r") as file:
        c_code = file.read()

    # Extract all function definitions
    function_pattern = r"void\s+(\w+)\s*\(([\s\S]*?)\)\s*\{([\s\S]*?)(?=void|\Z)"
    functions = re.finditer(function_pattern, c_code)

    function_dict = {}
    for match in functions:
        func_name = match.group(1)
        params_str = match.group(2)
        body_str = match.group(3)

        # Extract parameter names
        param_pattern = r"(?:[\w:]+\s+)+(\w+)(?:\s*\[.*?\])*"
        param_matches = re.finditer(param_pattern, params_str)
        params = [match.group(1) for match in param_matches]

        function_dict[func_name] = {"params": params, "body": body_str}

    # Check if top function exists
    if top_function_name not in function_dict:
        return ""

    # Find all function calls in the top function
    top_func_body = function_dict[top_function_name]["body"]
    call_pattern = r"(\w+)\((.*?)\);"
    calls = list(re.finditer(call_pattern, top_func_body))

    if not calls:
        # No function calls, find the parameter that's written to
        params = function_dict[top_function_name]["params"]
        written_params = []
        for param in params:
            assignment_pattern = rf"{param}\s*\[.*?\]\s*="
            if re.search(assignment_pattern, top_func_body):
                written_params.append(param)

        return written_params[-1] if written_params else ""
    else:
        # Get the last function call
        last_call = calls[-1]
        called_func = last_call.group(1)
        args_str = last_call.group(2)

        # Extract arguments
        args = [arg.strip() for arg in args_str.split(",")]

        # If the called function exists in our dictionary, map its parameters
        if called_func in function_dict:
            called_params = function_dict[called_func]["params"]

            # Find which parameter is written to in the called function
            called_body = function_dict[called_func]["body"]
            for i, param in enumerate(called_params):
                if i < len(args):  # Make sure we don't go out of bounds
                    assignment_pattern = rf"{param}\s*\[.*?\]\s*="
                    if re.search(assignment_pattern, called_body):
                        # This is an output parameter, map it to the argument
                        arg = args[i]
                        # If the arg is a top-level parameter, it's our live-out variable
                        if arg in function_dict[top_function_name]["params"]:
                            return arg

            # If we get here, the last argument is likely the live-out
            return args[-1]
        else:
            # Fallback: assume the last argument is the live-out variable
            return args[-1]


def add_pocc_pragmas(file_path):
    """
    Inserts Pocc pragmas into the generated C code

    For a multi-function (composed) schedule, wrap the entire file in:
      #pragma pocc-region-start
      {contents...}
      #pragma pocc-region-end
    For a single-function schedule, insert pragmas inside the function body
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    func_defs = re.findall(
        r"^\s*(?:void|int|float|double)\s+\w+\s*\(.*?\)\s*\{", content, re.MULTILINE
    )
    if len(func_defs) > 1:
        new_content = (
            "#pragma pocc-region-start\n" + content + "\n#pragma pocc-region-end\n"
        )
    else:
        lines = content.splitlines(keepends=True)
        inserted_start = False
        for i, line in enumerate(lines):
            if (not inserted_start) and (") {" in line):
                lines.insert(i + 1, "#pragma pocc-region-start {\n")
                inserted_start = True
                break
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "}":
                lines.insert(i, "}\n#pragma pocc-region-end\n")
                break
        new_content = "".join(lines)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def replace_unsupported_types(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    updated_content = content.replace("int32_t", "int").replace("int64_t", "int")
    updated_content = updated_content.replace("(float)", "")
    updated_content = re.sub(r"ap_int<\d{1,2}>", "int", updated_content)

    # rewrite bitshift operation
    updated_content = re.sub(
        r"(\w+\s*(?:\[[^\]]+\])?)\s*\*\s*2\b", r"\1 << 1", updated_content
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
