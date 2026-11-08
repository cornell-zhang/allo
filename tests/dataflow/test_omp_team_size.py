# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test: the simulator must not deadlock when a region has more
kernel instances than OMP_NUM_THREADS.

`_inject_omp_parallel_sections` wraps the PE calls in an `omp.parallel` >
`omp.sections`. A PE blocked on a stream spins inside its section, so if the
OpenMP team is smaller than the section count the runtime never starts the
sections that would unblock it and the region hangs forever -- silently, with
no indication of which process is blocked on which channel. Deep FIFOs mask it
by letting producers run to completion before anyone has to block, so it
presents as a design bug rather than a simulator one.

The region below is a 16-stage relay chain over depth-1 streams: every stage
must block, so the deadlock is immediate and does not depend on timing. Run
under OMP_NUM_THREADS=2, it hangs forever without the fix and finishes in
about two seconds with it.

The region runs in a subprocess with a timeout, because a test that hangs is
worse than a test that fails.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

# More stages than the OMP_NUM_THREADS the child runs under (2).
P = 16
NELEM = 8
# Generous: the region takes ~2s when it works, and only a hang is slower.
TIMEOUT_SEC = 120


def _build_and_run_chain():
    """A P-stage relay chain over depth-1 streams. Every stage must block."""

    @df.region()
    def chain(A: int32[NELEM], B: int32[NELEM]):
        # Depth 1: a producer cannot run ahead, so each stage really blocks.
        fifo: Stream[int32, 1][P]

        @df.kernel(mapping=[P], args=[A, B])
        def relay(local_A: int32[NELEM], local_B: int32[NELEM]):
            p = df.get_pid()
            with allo.meta_if(p == 0):
                for i in range(NELEM):
                    fifo[1].put(local_A[i])
            with allo.meta_elif(p == P - 1):
                for i in range(NELEM):
                    local_B[i] = fifo[p].get() + 1
            with allo.meta_else():
                for i in range(NELEM):
                    fifo[p + 1].put(fifo[p].get() + 1)

    A = np.arange(NELEM, dtype=np.int32)
    B = np.zeros(NELEM, dtype=np.int32)
    sim_mod = df.build(chain, target="simulator")
    sim_mod(A, B)
    # Each of the P-1 downstream stages adds one.
    np.testing.assert_array_equal(B, A + (P - 1))


def test_more_pes_than_omp_threads():
    """The chain must complete with an OpenMP team far smaller than the team
    the OpenMP default would pick."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--run-chain"],
            env=env,
            timeout=TIMEOUT_SEC,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the {P}-PE region did not finish in {TIMEOUT_SEC}s at "
            "OMP_NUM_THREADS=2: the OpenMP team is smaller than the section "
            "count, so a blocked PE's section is never unblocked. See "
            "_inject_omp_parallel_sections in allo/backend/simulator.py."
        )
    assert proc.returncode == 0, (
        f"the {P}-PE region failed at OMP_NUM_THREADS=2 "
        f"(exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )


if __name__ == "__main__":
    if "--run-chain" in sys.argv:
        _build_and_run_chain()
        sys.exit(0)
    test_more_pes_than_omp_threads()
    print("Dataflow Simulator Passed!")
