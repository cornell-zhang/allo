# Programmable Dataflow Architecture

## Problem
1. The only feasible scaling direction: scale-out architecture
- Some markets, e.g., AI datacenters/cars, justify interposer-based superchips up to the wafer-scale.
2. Dataflow grid (2D/3D) is a natural abstraction to model such system architecture. Current scope: single chip.
3. Design space of dataflow architecture?
- dataflow flexibility
- dataflow efficiency (power & area cost can be comparable to a PE; tolerable?)
    - different workload requirements: moes, dense/sparse gemm/gemv, graph/tree algs, physics/robot algs.
- {tiles x interconnects & protocols x global control plane}
    - we focus on the latter two.

## Background
1. What is the textbook-level wisdom?
- 