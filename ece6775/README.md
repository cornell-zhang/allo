# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ECE6775 - MoE and Attention+MoE Implementations

Allo implementations of MoE (Mixture of Experts) and Attention+MoE layers for FPGA acceleration.

## Structure

- `moe/`: Standalone MoE layer implementations
- `attention_moe/`: Attention + MoE combined implementations
- `llm_config/`: Shared configuration for different model sizes

## MoE Implementations

Each directory contains multiple versions:

- `*_base.py`: Manual implementation (no library functions)
- `*_lib.py`: Uses Allo library functions (nn.linear2d, nn.GeLU)
- `*_alt.py`: Optimized version with fused GeLU and row-level dataflow
- `pytorch_*.py`: PyTorch reference implementation

## Usage

Run any implementation directly:

```bash
python moe/allo_moe_alt.py
python attention_moe/allo_attention_moe_alt.py
```

## Configuration

Edit `llm_config/llm_config.py` to change model configuration. Default: `switch_base_8_scaled_1_8`

Available configs:
- `switch_base_8*`: Google Switch-Base-8 variants
- `mixtral_8x7b*`: Mixtral-8x7B variants
- `deepseek*`: DeepSeek MoE variants
- `custom`: Small test configuration

## Notes

- Small numerical differences (~1e-4) vs PyTorch are normal due to accumulation order
- Set `MODE` in each file to control build target: `llvm`, `sw_emu`, `hw_emu`, `hw`, `csyn`

Github PR link: https://github.com/cornell-zhang/allo/pull/489