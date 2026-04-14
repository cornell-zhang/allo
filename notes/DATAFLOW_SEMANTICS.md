<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Dataflow Simulation Semantics: Allo vs HLS CSIM

This document clarifies the execution semantics of the three simulation/verification
modes available for Allo dataflow regions, and their implications for design validation.

---

## 1. The Three Execution Models

### 1.1 Allo Simulator (`target="simulator"`)

**Mechanism**: OpenMP parallel threads.

Each `@df.kernel` becomes a C++ function called in a separate OpenMP thread.
All threads start simultaneously and communicate through lock-free FIFO queues
(implemented in `allo/backend/simulator.py`).

```
Thread 0: memory_tile_2x1()  ──────→ runs concurrently
Thread 1: compute_tile_2x1_0()  ──→ runs concurrently
Thread 2: compute_tile_2x1_1()  ──→ runs concurrently
```

**Stream semantics**:
- `put()`: **blocking** — waits (spin-poll) until FIFO has space
- `get()`: **blocking** — waits (spin-poll) until FIFO has data
- `try_put()`: **non-blocking** — returns immediately with success/fail
- `try_get()`: **non-blocking** — returns immediately with value+success
- `empty()`: **non-blocking** — reads current head/tail counter atomically
- `full()`: **non-blocking** — reads current head/tail counter atomically

**Deadlock behavior**: Can deadlock if the protocol is incorrect (e.g., MT
waits for CT while CT waits for MT with no buffers). The OMP thread scheduler
does NOT magically resolve deadlocks.

**Use when**: Functional verification of any design — blocking, non-blocking,
or handshake protocols. This is the primary correctness oracle.

---

### 1.2 Vitis HLS C-Simulation (`target="vitis_hls", mode="csim"`)

**Mechanism**: Sequential function calls within a single thread.

HLS CSIM compiles the HLS C++ kernel code natively (with `g++` + HLS headers).
The top-level wrapper calls dataflow processes **in source-code order**:

```
// HLS CSIM execution order (sequential):
entry_proc();         // 1st
memory_tile();        // 2nd ← runs to completion before compute_tile starts
compute_tile_0();     // 3rd
compute_tile_1();     // 4th
store_res();          // 5th
```

> **Reference**: Vitis HLS UG1399: "C simulation runs the testbench and the
> C model of the DUT. The dataflow processes are executed sequentially."

**Stream semantics** (`hls::stream<T>` in CSIM mode):
- `stream.write(v)`: If FIFO is **full** → **blocks** (spin-waits in a while loop)
- `stream.read()`: If FIFO is **empty** → **blocks**
- `stream.write_nb(v)`: Returns false immediately if full (no block)
- `stream.read_nb(v)`: Returns false immediately if empty (no block)
- `stream.empty()`: Returns current empty status (no side effects)
- `stream.full()`: Returns current full status (no side effects)

**Critical implication**: Because processes run sequentially, a design that
works correctly depends on **FIFO buffering being sufficient** to decouple
the producer from the consumer within a single sequential pass.

**What works in CSIM**:
```python
# WORKS: Producer fills FIFO fully, then consumer drains it
@df.kernel(mapping=[1])
def producer():
    for i in range(4):          # FIFO depth=4 → all fit without blocking
        S[0].put(i * 10)

@df.kernel(mapping=[1], args=[out])
def consumer(out):
    for i in range(4):          # Runs AFTER producer; FIFO has 4 items
        out[i] = S[0].get()
```

```python
# ALSO WORKS: Non-blocking variant with sufficient depth
@df.kernel(mapping=[1])
def producer_nb():
    for i in range(4):
        while not S[0].try_put(i * 10):  # Succeeds on 1st try (FIFO not full)
            pass
```

**What DEADLOCKS in CSIM**:
```python
# DEADLOCKS: Bidirectional handshake requires concurrency
@df.kernel(mapping=[1], args=[...])
def memory_tile(...):
    req_valid[0].try_put(MSG_WRITE)         # Sends request (FIFO has space ✓)
    while not grant_ready[0].try_get():     # Waits for grant ← SPINS FOREVER
        pass                                # CT hasn't run yet!

@df.kernel(mapping=[1])
def compute_tile():
    # This runs AFTER memory_tile completes (but MT is stuck in spin-wait)
    if not req_valid[0].empty():
        ...
        grant_ready[0].try_put(MSG_GRANT)   # Never reached!
```

**Use when**: Verifying **one-directional stream designs** where producer fills
FIFO(s) before consumer runs. Good for: data pipelines, systolic arrays with
fixed communication patterns, feedforward designs.

**NOT for**: Any design with bidirectional synchronization (request/grant,
credit-based, handshake protocols) — use the Allo simulator or RTL cosim instead.

---

### 1.3 Vitis HLS RTL Co-Simulation (`target="vitis_hls", mode="cosim"`)

**Mechanism**: RTL simulation (Xsim) with true concurrent processes.

After synthesis, the generated Verilog/VHDL is simulated with the C++ testbench
driving the top-level AXI ports. RTL processes (including each dataflow kernel)
run as concurrent always-blocks in the Verilog simulation.

**Stream semantics**: Cycle-accurate handshake (AXI-Stream or FIFO protocol).
Every `hls::stream.write()` and `.read()` maps to RTL control signals. True
concurrency is maintained.

**Use when**: Cycle-accurate functional verification after synthesis. Validates
that the RTL behaves identically to the C model. Supports ALL protocols including
bidirectional handshake.

**Cost**: ~10-100× slower than CSIM. Requires successful synthesis first.

---

## 2. Decision Matrix

| Design Pattern | Allo Simulator | HLS CSIM | HLS RTL Cosim |
|----------------|----------------|----------|---------------|
| Blocking FIFO (put/get) | ✓ | ✓ | ✓ |
| Non-blocking (try_put/try_get), one-directional | ✓ | ✓ | ✓ |
| Non-blocking with spin-wait, one-directional | ✓ | ✓ (if FIFO depth ≥ burst) | ✓ |
| Bidirectional handshake (request + grant) | ✓ | ✗ DEADLOCK | ✓ |
| Credit-based flow control | ✓ | ✗ DEADLOCK | ✓ |
| Stateful kernels (@ stateful) | ✓ | ✓ | ✓ |
| Sub-region calls (nested dataflow) | ✓ | ✓ | ✓ |

---

## 3. Semantics of HLS Dataflow in Hardware (CSYN)

In actual hardware (post-synthesis), each `@df.kernel` becomes a process
(FSM + datapath) running in a **pipelined dataflow** region:

- All processes start simultaneously when the top-level AXI ap_start fires
- Processes communicate through **physical FIFOs** (shift-register or BRAM)
- Blocking `.write()`/`.read()` insert handshake stall logic into the FSM
- Non-blocking `.write_nb()`/`.read_nb()` return immediately with a success bit

The dataflow pipeline achieves initiation interval (II) equal to the **bottleneck
process latency**. For designs with spin-wait loops (`while ok == 0`), HLS cannot
statically determine II → reports "undef". This is not an error; at runtime the
loop terminates when the condition is met (bounded by FIFO occupancy).

**Important HLS dataflow constraints**:
1. Every internal FIFO must have **exactly one producer process** and **exactly
   one consumer process** — no multicast, no single-process write-only streams
2. Each process must be invoked by `#pragma HLS dataflow` in a `for`-loop or
   called as a function in a dataflow-annotated scope
3. Processes with `ap_ctrl_none` (no done signal) and rewinding loops may
   trigger HLS 200-656 deadlock warning — this is expected for spin-wait designs

---

## 4. Allo-Specific Notes

### `@ stateful` variables

```python
spad: float32[BURST_SIZE] @ stateful = 0.0
```

In the **Allo simulator**: The variable lives in a heap-allocated array that
persists across repeated calls to `df.build()(...)`. OpenMP threads share the
process address space, so static state is naturally persistent.

In **HLS CSIM**: Lowered to a C++ `static` local variable inside the kernel
function. Persists across sequential calls to the kernel.

In **HLS RTL** (synthesis): Lowered to a `static` local → register array
(FF-based for small sizes ≤ ~32 elements, BRAM for larger). Persists between
top-level invocations (kernel calls).

### `df.get_pid()` with `mapping=[N]`

Each `@df.kernel(mapping=[N])` instance gets a **compile-time-constant** pid:
the Allo compiler generates `N` specialized kernel instances at synthesis time.
Stream indexing `stream[pid]` is resolved statically — equivalent to manually
unrolling.

In the **Allo simulator**: `get_pid()` returns the OpenMP thread index. Stream
array indexing is dynamic at runtime but each thread has exclusive access to its
slice.

### Simulator vs CSIM for `top_decoupled_2x1`

```
Allo simulator: PASS  ✓  (concurrent OMP threads, correct handshake)
HLS CSIM:       DEADLOCK (sequential, MT waits for CT that hasn't run)
HLS CSYN:       PASS  ✓  (synthesizes correctly at 411 MHz)
```

This demonstrates that **the Allo simulator is the correct tool for verifying
decoupled/handshaked designs** before submitting to HLS synthesis.

---

## 5. Recommendation

For the mesh accelerator project:

1. **Develop and debug** with `target="simulator"` — fast iteration, correct concurrency
2. **Verify synthesizability** with `target="vitis_hls", mode="csyn"` — gets LUT/FF/II
3. **Cycle-accurate validation** with `mode="cosim"` after synthesis — for timing-critical paths
4. **Avoid HLS CSIM** for any design with bidirectional stream dependencies
