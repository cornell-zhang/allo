<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Attention Kernels in Allo

This folder contains two FPGA implementations of multi-head attention written
in Allo. Both compute softmax attention without ever building the full score
matrix, but they organize the hardware in very different ways. The first is a
tiled FlashAttention engine. The second is a systolic array that uses
low-precision arithmetic.

## FlashAttention (`flash_Atten.py`)

This kernel follows the FlashAttention algorithm. For each head, the sequence
is split into small tiles of queries and keys. The kernel loads a query tile
and then streams the key and value tiles past it. For each key/value tile, it
computes the scores against the current query tile and updates a running
softmax for every query row: the largest score seen so far, the sum of
exponentials, and a partial output. When the maximum grows, the older partial
results are rescaled so they stay consistent. After every key tile has been
processed, the output is normalized once and written back. All arithmetic is
in single-precision floating point, so the result matches standard attention.

In hardware, this is a single engine that repeatedly loads tiles into on-chip
buffers, computes on them, and writes results out. Scheduling directives
unroll the inner computation within a tile, including the score calculation,
the exponentials, and the weighted sum over values, into parallel logic, and
they pipeline the memory transfers. Loading and computing happen one after the
other rather than overlapping. The design is a clear, compact baseline.

Our allo version is about 3 times faster than C++ HLS version while using much more resources like DSP, LUT, FF.

## Fused MHA Systolic Array (`fused_MHA_systolic.py`)

This kernel maps attention onto a two-dimensional grid of processing elements
built with Allo's dataflow interface. It uses a mixed-precision scheme: queries
and keys are quantized to 8-bit integers for the score computation, while
softmax and the multiplication by values stay in floating point. Before
quantization, the kernel subtracts the average key from every key. Softmax
cancels that shift, and removing it lets 8-bit integers represent the keys
more accurately.

Data flows through the grid in two directions. Each row of processing elements
is responsible for one query, which enters from the left together with an
empty running softmax state. Each column is responsible for one block of keys
and values, which enter from the top. As a query moves from left to right, each
processing element it passes compares the query against its block of keys,
updates the running softmax state, and hands the query and state to its right
neighbor. Keys and values move downward so that every row sees them. When a
query leaves the right edge of the grid, it has seen every key, and its state
already holds the final output. The processing elements on the edges of the
grid feed data in and drain it out, while dedicated load and store stages
connect the grid to memory.

Many queries are processed in parallel across rows, and successive groups of
queries follow one another through the grid as a wavefront. Every stage stays
busy, and communication stays local between neighboring elements.

The fused MHA systolic array is about 2 times faster than the first version. 
