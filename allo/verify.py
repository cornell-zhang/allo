import os
import past

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

    add_pocc_pragmas(prog_a_path)
    add_pocc_pragmas(prog_b_path)


    is_equivalent = past.verify(prog_a_path, prog_b_path, "v2")

    return is_equivalent


def add_pocc_pragmas(file_path):
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
