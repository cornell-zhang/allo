from air.mlir.ir import *
import air.mlir.passmanager
from air.dialects import air as airdialect
import air.mlir._mlir_libs._airMlir
import air.mlir._mlir_libs._airMlir.runner as _runner

import logging

# Configure the logging settings
logging.basicConfig(level=logging.DEBUG,  # Set the logging level (DEBUG, logger.info, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger(__name__)

import json
import tempfile
import os
import argparse

LINALG_TENSOR_TO_MEMREF_PIPELINE = "builtin.module(" + ",".join([
    # Bufferize.
    "func.func(scf-bufferize)",
    "func.func(linalg-bufferize)", "cse",
    "func-bufferize",
    "arith-bufferize",
    "func.func(tensor-bufferize)",
    "func.func(finalizing-bufferize)",
    "canonicalize",
    "cse"
]) + ")"

def _convert_module(module):
    if not isinstance(module, air.mlir.ir.Module):
        air_module = air.mlir.ir.Module.parse(str(module),air.mlir.ir.Context())
    else:
        air_module = module
    return air_module

class CostModel:
    def __init__(self):
        pass

    def op_stats(self, module):
        """Return operation count logger.information as JSON"""
        air_module = _convert_module(module)
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            name = tmpfile.name
            with air_module.context:
                pipeline = f"builtin.module(air-linalg-op-stats{{outputfile={name}}})"
                pm = air.mlir.passmanager.PassManager.parse(pipeline)
                pm.run(air_module)
            stats = open(name).read()
            os.unlink(name)
        return stats

class Runner:
    def __init__(self, json_model, trace_filename=None, sim_granularity="herd", verbose=False):
        self.json_model = json_model
        self.trace_filename = trace_filename
        self.sim_granularity = sim_granularity
        self.verbose = verbose

    def run(self, module, function):
        air_module = _convert_module(module)

        trace_tmpfile = None
        trace_filename = self.trace_filename
        if trace_filename is None:
            trace_tmpfile = tempfile.NamedTemporaryFile(delete=False)
            trace_filename = trace_tmpfile.name
        
        # the json model can be:
        #  1. json in string form
        #  2. json in python object form
        #  3. the name of a file containing (1)
        json_model = self.json_model
        if type(json_model) == str:
            if '.json' in json_model:
                with open(json_model) as f:
                    json_model = json.loads(f.read())
            else:
                json_model = json.loads(json_model)

        json_tmpfile = tempfile.NamedTemporaryFile(delete=False)
        json_tmpfile.write(str.encode(json.dumps(json_model)))
        json_tmpfile.close()

        _runner.run(air_module, json_tmpfile.name, trace_filename, function, self.sim_granularity, self.verbose)

        os.unlink(json_tmpfile.name)

        # return the trace and remove the temporary file
        # if the user didn't provide an output filename
        return_trace = None
        if trace_tmpfile:
            return_trace = open(trace_tmpfile.name).read()
            os.unlink(trace_tmpfile.name)

        return return_trace


def air_codegen_start_runner(mlir_file, top_func, arch_file, trace_file, project_dir):

    with air.mlir.ir.Context() as ctx, Location.unknown():
        airdialect.register_dialect(ctx)

        with open(mlir_file, "r") as f:
            air_module = air.mlir.ir.Module.parse(f.read(), ctx)
        
        # convert linalg on tensors to linalg on memrefs
        pm = air.mlir.passmanager.PassManager.parse(LINALG_TENSOR_TO_MEMREF_PIPELINE)
        pm.run(air_module.operation)

        with open(os.path.join(project_dir, "linalg.bufferized.mlir"), "w") as f:
            f.write(str(air_module))
        logger.info(f"Linalg bufferization result: {os.path.join(project_dir, 'linalg.bufferized.mlir')}")

        # tile and map to air
        pipeline = "builtin.module("+",".join([
            "air-linalg-codegen{l2-tile-size=64,64,64 l2-promote=true l1-tile-size=32,32,32 l1-promote=true}",
            "canonicalize", "cse",
            "air-par-to-herd{depth=1}",
            "air-copy-to-dma",
            "air-par-to-launch{has-air-segment=true}",
            "canonicalize", "cse",
        ])+')'
        pm = air.mlir.passmanager.PassManager.parse(pipeline)
        pm.run(air_module.operation)

        with open(os.path.join(project_dir, "air.herd.mlir"), "w") as f:
            f.write(str(air_module))
        logger.info(f"Air herd generation result: {os.path.join(project_dir, 'air.herd.mlir')}")

        # generate dependency logger.information for runner
        pipeline = "builtin.module("+",".join([
            "air-dependency",
            "air-dependency-schedule-opt",
            "air-specialize-dma-broadcast",
            "air-dma-to-channel",
            "canonicalize", "cse",
            "air-dependency-canonicalize",
            f"air-dependency-parse-graph{{output-dir={project_dir}/dot_graphs/}}",
            "canonicalize", "cse",
            "air-place-herds{num-rows=2 num-cols=2 row-anchor=0 col-anchor=0}",
            "air-label-scf-for-to-ping-pong",
            "air-ping-pong-transform"
        ])+')'
        pm = air.mlir.passmanager.PassManager.parse(pipeline)
        pm.run(air_module.operation)

        with open(os.path.join(project_dir, "air.async.mlir"), "w") as f:
            f.write(str(air_module))
        logger.info(f"Air async generation result: {os.path.join(project_dir, 'air.async.mlir')}")

        # load arch file
        with open(arch_file) as f:
            arch = json.loads(f.read())
        
        logger.info(f"Running on arch: {arch_file}")
        runner = Runner(arch, trace_file, "core")
        trace = runner.run(air_module, top_func)
        logger.info(f"Produced trace file: {os.path.abspath(trace_file)}")



def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file")
    parser.add_argument("--top_func", help="top functsion")
    parser.add_argument("--arch", help="arch file")
    parser.add_argument("--trace", help="trace file")
    parser.add_argument("--project_dir", help="project directory")
    
    args = parser.parse_args()
    # get input filename
    input_file = args.input
    # get arch filename
    arch_file = args.arch
    # get trace filename
    trace_file = args.trace
    # get project directory
    project_dir = args.project_dir
    # get top function
    top_func = args.top_func

    air_codegen_start_runner(input_file, top_func, arch_file, trace_file, project_dir)



if __name__ == "__main__":
    main()