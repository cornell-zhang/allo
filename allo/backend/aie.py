import os
import re
import io
import subprocess
import time

def run_process(cmd, pattern=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err:
        raise RuntimeError("Error raised: ", err.decode())
    if pattern:
        return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")


class AIEModule:
    def __init__(
        self,
        mod,
        top_func_name,
        mode="sim",
        project=None,
        arch_file=None
    ):
        self.mod = mod
        self.top_func_name = top_func_name
        self.mode = mode
        if project is None:
            project = f"air_prj.{int(time.time())}"
        self.project = project

        # default arch file
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        default_arch_file = os.path.join(curr_dir, "../harness/air_runner/vck190_arch.json")
        if arch_file is None:
            arch_file = default_arch_file

        # Make a project dir
        os.makedirs(self.project, exist_ok=True)
        self.project = os.path.abspath(self.project)
        arch_file_base = os.path.basename(arch_file)
        self.arch_file = os.path.join(self.project, arch_file_base)
        # copy arch file to project dir
        os.system(f"cp {default_arch_file} {self.project}")
        # copy air_runner script to project dir
        os.system(f"cp {curr_dir}/../harness/air_runner/air_runner.py {self.project}")


    def __repr__(self):
        return f"AIEModule({self.top_func_name}, {self.mode}, {self.project})"


    def __call__(self):
        if self.mode == "sim":
            cmd = "cd " + self.project + "; python3 air_runner.py"
            # write self.mod to a file
            with open(os.path.join(self.project, "input.mlir"), "w") as f:
                f.write(str(self.mod))
            cmd += " input.mlir"
            cmd += " --top_func " + self.top_func_name
            cmd += " --arch " + self.arch_file
            cmd += " --trace ./trace.out"
            cmd += " --project_dir " + self.project

            print("running cmd: ", cmd)
            run_process(cmd)

        else: 
            raise RuntimeError("not implemented")
        
        