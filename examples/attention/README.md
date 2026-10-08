<!-- Copyright Allo authors. All Rights Reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Attention Kernels in Allo

This directory contains two FPGA implementations of multi-head attention in
Allo. Both use online softmax, so they never materialize the full attention
score matrix. They differ in dataflow, precision, and the amount of spatial
parallelism they expose.

## Tiled FlashAttention (`./flash_attention/flash_attention.py`)

This is a compact, single-engine FlashAttention baseline. For each head, it
keeps a tile of queries on chip and streams tiles of keys and values past it.
For every key/value tile, the engine updates each query row's running maximum,
exponential sum, and partial output. It normalizes and stores the output only
after all key/value tiles have been processed.

The kernel uses FP32 arithmetic and unrolls the compute within a tile. Tile
loads and computation are pipelined, but they are not overlapped. In the
reported implementation, the Allo version is about 3x faster than the C++ HLS
version, at the cost of substantially greater FPGA resource use.

## Fused MHA Systolic Array (`./fused_mha_systolic/fused_mha_systolic.py`)

This design maps attention to a two-dimensional grid of processing elements
(PEs). Queries and keys use INT8 for score computation; softmax and the
value-weighted accumulation remain in floating point. Before quantization,
the mean key is subtracted from every key. This shift cancels in softmax while
improving the effective INT8 key range.

Queries and their running softmax state travel from left to right. Key/value
blocks travel from top to bottom. At each PE, a query is scored against the
local key block, its online-softmax state is updated, and the result moves to
the next PE. Multiple query rows advance through the array as a wavefront, so
communication is local and the array can process several queries at once.

## Results from the supplied runs

Resource values are the reported HLS estimates; latency values are the XRT
kernel wall-clock measurements shown in the supplied logs.

| Metric | Fused MHA systolic array | Tiled FlashAttention engine |
| --- | ---: | ---: |
| BRAM | 2,196 (54%) | 100 (2%) |
| DSP | 5,508 (61%) | 572 (6%) |
| FF | 771,964 (29%) | 75,119 (2%) |
| LUT | 877,146 (67%) | 52,075 (3%) |
| URAM | Not used | Not used |
| Measured kernel wall-clock time |  182,654ns (0.183 ms) | 391,748 ns (0.392 ms) |
