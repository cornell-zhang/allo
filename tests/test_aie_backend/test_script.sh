#!/bin/bash

mkdir -p air_prj

python /home/nz264/shared/allo/allo/harness/air_runner/air_runner.py \
    ./sample_input.mlir \
    --top_func matmul \
    --trace ./trace.out \
    --project_dir ./air_prj 