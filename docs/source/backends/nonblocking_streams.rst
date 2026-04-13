.. Copyright Allo authors. All Rights Reserved.
.. SPDX-License-Identifier: Apache-2.0

Non-Blocking Stream Operations
================================

.. note::

   **Branch:** ``feature/mesh-accelerator`` (commits ``0a01930`` and ``01f25e2``).
   This feature is not yet merged into ``main``. This document tracks the implementation
   for the upcoming PR.

Overview
--------

Allo's dataflow programming model connects kernels via typed FIFO streams (``Stream[T, depth]``).
By default streams are **blocking**: ``put()`` stalls the producer when the FIFO is full, and
``get()`` stalls the consumer when it is empty.

Non-blocking operations expose the FIFO's full/empty status and return a success flag instead
of blocking, enabling:

- **Valid-ready handshake** protocols (AXI-S, NoC credit-based flow control)
- **Arbitration** — a single kernel polling *multiple* request channels without deadlocking
- **Decoupled memory/compute tiles** with explicit message-passing control

New API
-------

Four methods were added to the ``Stream`` type (``allo/ir/types.py``, lines 349–360):

.. code-block:: python

   # allo/ir/types.py (feature/mesh-accelerator)

   class Stream:
       def put(self, data): ...          # blocking write (existing)
       def get(self)       -> T: ...     # blocking read  (existing)

       def try_put(self, data) -> int1:  # non-blocking write; returns 1 on success
       def try_get(self)  -> (T, int1):  # non-blocking read; returns (data, success)
       def empty(self)    -> int1:       # True when FIFO has no items
       def full(self)     -> int1:       # True when FIFO is at capacity

Typical usage pattern (spin-until-success):

.. code-block:: python

   # Producer: spin until FIFO accepts the token
   sent: int1 = 0
   while sent == 0:
       sent = req_valid[ct_id].try_put(MSG_WRITE)

   # Consumer: poll before committing to a read
   if not req_valid[id].empty():
       msg_type, has_req = req_valid[id].try_get()

Why Non-Blocking Is Necessary for Arbitration
----------------------------------------------

Consider a **many-to-one** topology: two Memory Tiles (MT0, MT1) share one Compute Tile (CT).
With *blocking* ``get()``, if CT blocks waiting for MT0 while MT1 sends first, the system
deadlocks permanently.

Non-blocking polling resolves this:

.. code-block:: python

   while served < N_MTS:
       if not req_from_mt[0].empty():
           msg, ok = req_from_mt[0].try_get()
           if ok: handle(0)
       if not req_from_mt[1].empty():
           msg, ok = req_from_mt[1].try_get()
           if ok: handle(1)

Working Examples
----------------

All examples live in the ``tests/dataflow/`` directory on the ``feature/mesh-accelerator`` branch.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Description
   * - ``tests/dataflow/test_stream_nb_simple.py``
     - Unit tests for ``try_put`` / ``try_get`` / ``empty`` / ``full`` with the Allo simulator.
       Six self-contained test cases, each exercising one primitive. **Start here.**
   * - ``tests/dataflow/test_decoupled_mesh.py``
     - End-to-end message-passing tests: 1-CT single-handshake protocol and 2×1 decoupled mesh.
       Demonstrates the full valid-ready handshake pattern.
   * - ``tests/dataflow/test_stream_ops_ir.py``
     - IR-level checks — verifies that the four new MLIR ops are emitted correctly.
   * - ``tests/dataflow/test_stream_ops_sim.py``
     - Simulator-level round-trip tests (producer → consumer via ``try_put`` / ``try_get``).
   * - ``tests/dataflow/test_stream_ops_hls.py``
     - HLS codegen checks — verifies ``nb_write()`` / ``nb_read()`` / ``.empty()`` appear in the
       emitted Vivado HLS C++ and TAPA C++.

Running the Simulator Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

No HLS tool required:

.. code-block:: bash

   conda activate allo
   python -m pytest tests/dataflow/test_stream_nb_simple.py     -v
   python -m pytest tests/dataflow/test_decoupled_mesh.py       -v
   python -m pytest tests/dataflow/test_stream_ops_sim.py       -v

Running the HLS Codegen Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requires Vitis HLS (``brg-zhang-xcel`` server):

.. code-block:: bash

   conda activate allo
   python -m pytest tests/dataflow/test_stream_ops_hls.py       -v

Implementation Internals
------------------------

The implementation touches four layers of the compiler stack.

Layer 1 — Python frontend (``allo/ir/types.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Stream.try_put`` / ``try_get`` / ``empty`` / ``full`` are stub methods that serve as
syntactic targets for the AST builder. They have no runtime Python logic; the builder
intercepts them and emits MLIR operations instead.

**Location:** ``allo/ir/types.py``, lines 349–360 (branch ``feature/mesh-accelerator``).


Layer 2 — MLIR operations (``mlir/include/allo/Dialect/AlloOps.td``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two new operations were added to the Allo MLIR dialect:

``allo.stream_try_put``
   Takes ``(stream, indices, data)``; returns ``i1`` success flag.

``allo.stream_try_get``
   Takes ``(stream, indices)``; returns ``(data, i1 success_flag)`` — a two-result op.

These are defined in ``mlir/include/allo/Dialect/AlloOps.td`` and the generated Python
bindings appear in
``mlir/build/include/allo/Bindings/dialects/_allo_ops_gen.py``
(``StreamTryPutOp``, ``StreamTryGetOp``, lines 2287–2378).

Layer 3 — AST builder (``allo/ir/builder.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The builder's ``visit_Call`` handler was extended to recognise the four new method names
on ``Stream`` objects and emit the corresponding MLIR ops:

**Location:** ``allo/ir/builder.py``, lines 2622–2676 (branch ``feature/mesh-accelerator``).

Key lowering details:

- ``try_put`` emits ``allo.stream_try_put``; the ``i1`` result is unpacked as the return value.
- ``try_get`` emits ``allo.stream_try_get``; the two results ``(data, success)`` are tuple-bound
  and unpacked at the call site via Python destructuring (``val, ok = stream.try_get()``).
- ``empty`` / ``full`` are lowered to ``allo.stream_try_get`` / ``allo.stream_try_put`` with a
  *probe-only* mode (no data consumed/produced).

Layer 4a — Simulator backend (``allo/backend/simulator.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OpenMP-threaded simulator was extended to:

1. **Interpret** ``StreamTryPutOp`` and ``StreamTryGetOp`` by inspecting the head/tail pointers
   of the shared FIFO without blocking.
2. **Flush** shared pointers with ``openmp_d.FlushOp`` before and after every non-blocking
   access to prevent stale cache lines across OpenMP threads.
3. **Recurse** ``_process_function_streams`` into nested kernel calls (fixing a pre-existing
   deadlock for hierarchical dataflow regions).

Layer 4b — HLS backends
~~~~~~~~~~~~~~~~~~~~~~~~

**Vivado HLS** (``mlir/lib/Translation/EmitVivadoHLS.cpp``):

.. code-block:: cpp

   // try_put → hls::stream::write_nb()
   os << streamName << ".write_nb(" << dataArg << ")";

   // try_get → hls::stream::read_nb()
   os << streamName << ".read_nb(" << dataArg << ")";

   // empty / full → hls::stream::empty() / full()
   os << streamName << ".empty()";

**TAPA** (``mlir/lib/Translation/EmitTapaHLS.cpp``):

.. code-block:: cpp

   // try_put → tapa::ostream::try_write()
   os << streamName << ".try_write(" << dataArg << ")";

   // try_get → tapa::istream::try_read()
   os << streamName << ".try_read(" << dataArg << ")";

**Catapult HLS** (``mlir/lib/Translation/EmitCatapultHLS.cpp``):

.. code-block:: cpp

   // try_put → ac_channel::nb_write()
   os << streamName << ".nb_write(" << dataArg << ")";

   // try_get → ac_channel::nb_read()
   os << streamName << ".nb_read(" << dataArg << ")";

   // empty → ac_channel::empty()
   os << streamName << ".empty()";

HLS Synthesis Cost (Vitis HLS, U280, estimated)
-----------------------------------------------

Based on synthesis experiments in ``tests/dataflow/hls_synth_streams.py``:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Design
     - LUT
     - FF
     - II
   * - Blocking stream (FIFO depth 2)
     - 1,417
     - 248
     - 7
   * - Non-blocking stream (try_put/try_get)
     - 1,457 (+2.8%)
     - 260 (+4.8%)
     - 7

Non-blocking operations add only ~O(1) LUTs per stream (one comparator + branch).
The success bit propagation is the main scheduling concern in HLS — keep it in control
logic, not on the datapath critical path.

Files Modified (PR Scope)
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - File
     - Change
   * - ``allo/ir/types.py``
     - Add ``try_put``, ``try_get``, ``empty``, ``full`` stubs to ``Stream``
   * - ``allo/ir/builder.py``
     - Lower the four new methods to MLIR ops in ``visit_Call``
   * - ``allo/ir/infer.py``
     - Type-inference rules for ``StreamTryPutOp`` / ``StreamTryGetOp``
   * - ``mlir/include/allo/Dialect/AlloOps.td``
     - Define ``allo.stream_try_put`` and ``allo.stream_try_get`` ops
   * - ``mlir/include/allo/Translation/EmitBaseHLS.h``
     - Declare virtual emitter hooks for non-blocking ops
   * - ``mlir/lib/Translation/EmitVivadoHLS.cpp``
     - Vivado HLS: emit ``write_nb`` / ``read_nb`` / ``empty`` / ``full``
   * - ``mlir/lib/Translation/EmitTapaHLS.cpp``
     - TAPA: emit ``try_write`` / ``try_read`` / ``empty``
   * - ``allo/backend/simulator.py``
     - Interpret non-blocking ops; OMP flush; recursive kernel injection
   * - ``tests/dataflow/test_stream_nb_simple.py``
     - Simulator unit tests (3 test cases)
   * - ``tests/dataflow/test_decoupled_mesh.py``
     - End-to-end 1-CT and 2×1 mesh tests
   * - ``tests/dataflow/test_stream_ops_ir.py``
     - MLIR op emission tests
   * - ``tests/dataflow/test_stream_ops_sim.py``
     - Simulator round-trip tests
   * - ``tests/dataflow/test_stream_ops_hls.py``
     - HLS codegen string-match tests

See Also
--------

- ``docs/source/dive/dataflow.rst`` — Dataflow programming model overview
