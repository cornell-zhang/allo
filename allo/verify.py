import os
import past
import re

def verify(schedule_a, schedule_b):
    '''
    Run PAST verifier on the two schedules, returns whether they are equivalent
    Currently only supports single function, two-input + one-output schedules 
    '''
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

    add_pocc_pragmas_single_func(prog_a_path)
    add_pocc_pragmas_single_func(prog_b_path)

    replace_unsupported_types(prog_a_path)
    replace_unsupported_types(prog_b_path)

    output_var = "v0"

    is_equivalent = past.verify(prog_a_path, prog_b_path, output_var)

    return is_equivalent


def add_pocc_pragmas_single_func(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # insert `#pragma pocc-region-start {` after the first function header and opening brace
    for i, line in enumerate(lines):
        if ") {" in line:
            lines.insert(i + 1, "#pragma pocc-region-start {\n")
            break

    # insert `#pragma pocc-region-end` before the final closing brace
    for i in range(len(lines) - 1, -1, -1):
        if "}" in lines[i]:
            lines.insert(i, "}\n#pragma pocc-region-end\n")
            break

    with open(file_path, "w") as f:
        f.writelines(lines)


def replace_unsupported_types(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    # replace types and type conversions
    updated_content = content.replace("int32_t", "int").replace("int64_t", "int")
    updated_content = updated_content.replace("(float)", "")
    updated_content = re.sub(r"ap_int<\d{1,2}>", "int", updated_content)

    with open(file_path, 'w') as f:
            f.write(updated_content)