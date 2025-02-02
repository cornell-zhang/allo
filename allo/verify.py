import os
import re
import past

def verify(schedule_a, schedule_b):
    """
    Run PAST verifier on the two schedules, returns whether they are equivalent
    """
    temp_dir = 'tmp'
    os.makedirs(temp_dir, exist_ok=True)

    prog_a_path = os.path.join(temp_dir, "a.c")
    prog_b_path = os.path.join(temp_dir, "b.c")

    mod_a = schedule_a.build(target="vhls")
    mod_b = schedule_b.build(target="vhls")

    with open(prog_a_path, "w") as f:
        f.write(str(mod_a))
    with open(prog_b_path, "w") as f:
        f.write(str(mod_b))

    add_pocc_pragmas(prog_a_path)
    add_pocc_pragmas(prog_b_path)
    replace_unsupported_types(prog_a_path)
    replace_unsupported_types(prog_b_path)

    # detect output vars
    out_a = get_output_var_from_file(prog_a_path)
    out_b = get_output_var_from_file(prog_b_path)

    # if output vars have different names, rename program B's output var
    if out_a != out_b:
        rewrite_output_variable(prog_b_path, out_b, out_a)
        out_b = out_a

    is_equivalent = past.verify(prog_a_path, prog_b_path, out_a)
    return is_equivalent


def rewrite_output_variable(file_path, old_var, new_var):
    """
    Rewrite the file at file_path by replacing all whole–word occurrences of old_var with new_var,
    but (if possible) only inside the call block—the region between
    "#pragma pocc-region-start" and "#pragma pocc-region-end".
    """
    with open(file_path, "r") as f:
        content = f.read()
    if "#pragma pocc-region-start" in content and "#pragma pocc-region-end" in content:
        # only rewrite call region
        pattern = re.compile(
            r"(#pragma\s+pocc-region-start\s*(?:\{)?\s*)(.*?)(\s*(?:\})?\s*#pragma\s+pocc-region-end)",
            re.DOTALL
        )
        def repl(match):
            start, block, end = match.groups()
            new_block = re.sub(r"\b" + re.escape(old_var) + r"\b", new_var, block)
            return start + new_block + end
        new_content = pattern.sub(repl, content)
    else:
        # global rewrite: not in call region
        new_content = re.sub(r"\b" + re.escape(old_var) + r"\b", new_var, content)
    with open(file_path, "w") as f:
        f.write(new_content)


def get_output_var_from_file(file_path):
    """
    Attempt to extract the output (live-out) variable from the generated code
    
    1. First, search for a return statement
    2. If no return is found, look for a call region wrapped by
       "#pragma pocc-region-start" and "#pragma pocc-region-end" and take the last argument
       of the final function call
    3. If all else fails, default to "v0"
    """
    with open(file_path, "r") as f:
        content = f.read()

    # look for return statement
    m = re.search(r"\breturn\s+\(?\s*([a-zA-Z_]\w*)\s*\)?\s*;", content)
    if m:
        return m.group(1).strip()

    # 2. look for call region
    m_region = re.search(
        r"#pragma\s+pocc-region-start\s*(?:\{)?\s*(.*?)\s*(?:\})?\s*#pragma\s+pocc-region-end",
        content, re.DOTALL
    )
    if m_region:
        region = m_region.group(1)
        calls = re.findall(r"\b\w+\s*\(([^)]*)\)\s*;", region)
        if calls:
            last_call = calls[-1]
            args = last_call.split(',')
            if args:
                output_candidate = args[-1].strip()
                output_candidate = re.sub(r"[\)\s]+$", "", output_candidate)
                return output_candidate

    # default if no output var is detected
    return "v0"


def add_pocc_pragmas(file_path):
    """
    Inserts Pocc pragmas into the generated C code
    
    For a multi–function (composed) schedule, wrap the entire file in:
      #pragma pocc-region-start 
      {contents...}
      #pragma pocc-region-end
    For a single–function schedule, insert pragmas inside the function body
    """
    with open(file_path, "r") as f:
        content = f.read()

    func_defs = re.findall(r"^\s*(?:void|int|float|double)\s+\w+\s*\(.*?\)\s*\{", content, re.MULTILINE)
    if len(func_defs) > 1:
        new_content = "#pragma pocc-region-start\n" + content + "\n#pragma pocc-region-end\n"
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

    with open(file_path, "w") as f:
        f.write(new_content)

def replace_unsupported_types(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    updated_content = content.replace("int32_t", "int").replace("int64_t", "int")
    updated_content = updated_content.replace("(float)", "")
    updated_content = re.sub(r"ap_int<\d{1,2}>", "int", updated_content)
    
    # rewrite bitshift operation
    updated_content = re.sub(r"(\w+\s*(?:\[[^\]]+\])?)\s*\*\s*2\b", r"\1 << 1", updated_content)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
