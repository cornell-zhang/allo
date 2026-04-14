# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import pytest
import allo
from allo.ir.types import int32, float32, int1, Stream, Stateful
import numpy as np
import allo.dataflow as df

BURST_SIZE = 16
N_CTS = 2           # Number of compute tiles in the 2x1 mesh

# Opcodes for the single-CT message-passing protocol
MSG_REQ_READ = 1
MSG_REQ_WRITE = 2
MSG_ACK = 3

# Opcodes for the hierarchical 2x1 decoupled mesh protocol
MSG_WRITE   = 1     # MT→CT: push data burst
MSG_READ    = 2     # MT→CT: request data burst
MSG_COMPUTE = 4     # MT→CT: trigger in-place computation (vadd +1)
MSG_GRANT   = 3     # CT→MT: acknowledge (same value as MSG_ACK for clarity)

@df.region()
def top_message_passing(
    base_addr: int32[1],
    in_payload: float32[BURST_SIZE],
    out_payload: float32[BURST_SIZE]
):
    # Handshaked Control Streams
    req_valid: Stream[int32, 2][1]
    req_addr: Stream[int32, 2][1]
    grant_ready: Stream[int32, 2][1]
    # Unconditional Data Burst Streams
    data_tx: Stream[float32, BURST_SIZE][1]
    data_rx: Stream[float32, BURST_SIZE][1]
    
    @df.kernel(mapping=[1], args=[base_addr, in_payload, out_payload])
    def memory_tile(
        b_addr: int32[1],
        in_p: float32[BURST_SIZE],
        out_p: float32[BURST_SIZE]
    ):
        # 1. MT initiates a WRITE request (Valid)
        req_sent: int1 = 0
        while req_sent == 0:
            # try_put models raising the "Valid" signal
            req_sent = req_valid[0].try_put(MSG_REQ_WRITE)
            if req_sent == 1:
                req_addr[0].put(b_addr[0])
                
        # 2. MT waits for a GRANT (Ready / Credit) from CT
        granted: int1 = 0
        while granted == 0:
            # try_get models polling for the "Ready" signal or "Credit"
            grant_msg, granted = grant_ready[0].try_get()
            if granted == 1 and grant_msg == MSG_ACK:
                # 3. Handshake complete! MT starts Unconditional Data Burst
                for i in range(BURST_SIZE):
                    data_tx[0].put(in_p[i])
                    
        # 4. MT initiates a READ request
        req_sent = 0
        while req_sent == 0:
            req_sent = req_valid[0].try_put(MSG_REQ_READ)
            if req_sent == 1:
                req_addr[0].put(b_addr[0])
                
        # 5. MT waits for a GRANT for reading, meaning CT has prepared the data
        granted = 0
        while granted == 0:
            grant_msg, granted = grant_ready[0].try_get()
            if granted == 1 and grant_msg == MSG_ACK:
                # 6. Handshake complete! MT receives Unconditional Data Burst
                for i in range(BURST_SIZE):
                    out_p[i] = data_rx[0].get()

    @df.kernel(mapping=[1])
    def compute_tile():
        data_mem: float32[256] @ Stateful = 0.0
        
        running: int1 = 1
        req_count: int32 = 0
        
        while running == 1 and req_count < 2:
            # 1. CT polls for incoming requests (Checking Valid signal)
            has_req: int1 = 0
            msg_type: int32 = 0
            
            if not req_valid[0].empty():
                msg_type, has_req = req_valid[0].try_get()
                
            if has_req == 1:
                addr: int32 = req_addr[0].get()
                
                if msg_type == MSG_REQ_WRITE:
                    # 2. CT grants the write request (Sending Ready/Credit)
                    grant_sent: int1 = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[0].try_put(MSG_ACK)
                        
                    # 3. CT receives Unconditional Data Burst
                    for i in range(BURST_SIZE):
                        data_mem[addr + i] = data_tx[0].get()
                        
                elif msg_type == MSG_REQ_READ:
                    # Modify data slightly to prove CT processed it
                    for i in range(BURST_SIZE):
                        data_mem[addr + i] = data_mem[addr + i] + 1.0
                        
                    # 2. CT grants the read request (Sending Ready/Credit)
                    grant_sent = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[0].try_put(MSG_ACK)
                        
                    # 3. CT sends Unconditional Data Burst
                    for i in range(BURST_SIZE):
                        data_rx[0].put(data_mem[addr + i])
                        
                req_count += 1
    
    pass

# ─── Hierarchical 2x1 Decoupled Mesh ─────────────────────────────────────────
#
# Architecture:
#   Memory Tile (MT)  ←valid-ready handshake→  Compute Tile 0 (CT0)
#                     ←valid-ready handshake→  Compute Tile 1 (CT1)
#
# Protocol per CT (3 phases):
#   1. WRITE  : MT sends N elements to CT scratchpad (handshake + burst)
#   2. COMPUTE: MT triggers in-place vadd (+1.0) on CT (handshake only)
#   3. READ   : MT reads N elements from CT (handshake + burst)
#
# Key advantage over blocking (top_2x1 in test_hierachical_mesh.py):
#   - MT dispatches WRITE to CT0 and CT1 independently (no NOP padding)
#   - CTs run concurrently and independently; no fixed-II constraint
#   - Decoupled control: MT can target any subset of CTs on demand

@df.region()
def top_decoupled_2x1(
    in0:  float32[BURST_SIZE],   # Input data for CT0
    in1:  float32[BURST_SIZE],   # Input data for CT1
    out0: float32[BURST_SIZE],   # Output data from CT0
    out1: float32[BURST_SIZE],   # Output data from CT1
):
    # Per-CT handshaked control streams (MT→CT: request; CT→MT: grant)
    req_valid:   Stream[int32, 2][N_CTS]
    grant_ready: Stream[int32, 2][N_CTS]
    # Per-CT unconditional data burst streams
    data_mt2ct: Stream[float32, BURST_SIZE][N_CTS]   # MT → CT data
    data_ct2mt: Stream[float32, BURST_SIZE][N_CTS]   # CT → MT data

    @df.kernel(mapping=[1], args=[in0, in1, out0, out1])
    def memory_tile_2x1(
        in0_p:  float32[BURST_SIZE],
        in1_p:  float32[BURST_SIZE],
        out0_p: float32[BURST_SIZE],
        out1_p: float32[BURST_SIZE],
    ):
        # ── Phase 1: WRITE to CT0 ──────────────────────────────────────────
        sent: int1 = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_WRITE)
        ack: int1 = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()
        for i in range(BURST_SIZE):
            data_mt2ct[0].put(in0_p[i])

        # ── Phase 1: WRITE to CT1 ──────────────────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[1].try_put(MSG_WRITE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[1].try_get()
        for i in range(BURST_SIZE):
            data_mt2ct[1].put(in1_p[i])

        # ── Phase 2: COMPUTE trigger to CT0 ───────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_COMPUTE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()

        # ── Phase 2: COMPUTE trigger to CT1 ───────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[1].try_put(MSG_COMPUTE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[1].try_get()

        # ── Phase 3: READ from CT0 ────────────────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_READ)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()
        for i in range(BURST_SIZE):
            out0_p[i] = data_ct2mt[0].get()

        # ── Phase 3: READ from CT1 ────────────────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[1].try_put(MSG_READ)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[1].try_get()
        for i in range(BURST_SIZE):
            out1_p[i] = data_ct2mt[1].get()

    @df.kernel(mapping=[N_CTS])
    def compute_tile_2x1():
        """Each CT instance handles one req_valid/grant_ready/data channel."""
        id = df.get_pid()   # 0 or 1 — resolved at compile time per instance
        spad: float32[BURST_SIZE] @ Stateful = 0.0

        req_count: int32 = 0
        while req_count < 3:   # Expects: WRITE, COMPUTE, READ
            has_req: int1 = 0
            msg_type: int32 = 0

            # Non-blocking poll (valid-ready: check if request present)
            if not req_valid[id].empty():
                msg_type, has_req = req_valid[id].try_get()

            if has_req == 1:
                if msg_type == MSG_WRITE:
                    # Grant the write, then receive burst data
                    grant_sent: int1 = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[id].try_put(MSG_GRANT)
                    for i in range(BURST_SIZE):
                        spad[i] = data_mt2ct[id].get()

                elif msg_type == MSG_COMPUTE:
                    # In-place computation: vadd +1.0 (simulates compute kernel)
                    for i in range(BURST_SIZE):
                        spad[i] = spad[i] + 1.0
                    # Send completion grant
                    grant_sent = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[id].try_put(MSG_GRANT)

                elif msg_type == MSG_READ:
                    # Grant the read, then send burst data
                    grant_sent = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[id].try_put(MSG_GRANT)
                    for i in range(BURST_SIZE):
                        data_ct2mt[id].put(spad[i])

                req_count += 1


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_decoupled_message_passing():
    print("Scalable Message-Passing Interconnect architecture parsed successfully")

    # Verify module building
    simulator = df.build(top_message_passing, target="simulator")

    # Test Data
    np_base_addr = np.array([0], dtype=np.int32)
    np_in_payload = np.arange(BURST_SIZE, dtype=np.float32)
    np_out_payload = np.zeros(BURST_SIZE, dtype=np.float32)

    # Run Simulation
    simulator(np_base_addr, np_in_payload, np_out_payload)

    # Verify that CT modified the payload (+1.0)
    np.testing.assert_allclose(np_out_payload, np_in_payload + 1.0)
    print("Simulation passed successfully!")


def test_decoupled_2x1_mesh():
    """
    2×1 hierarchical decoupled mesh: 1 MT + 2 independent CTs.
    Protocol: valid-ready handshake (try_put/try_get) for control,
    unconditional burst streaming for data.
    Both CTs run concurrently; MT dispatches independently.
    """
    sim = df.build(top_decoupled_2x1, target="simulator")

    np_in0  = np.arange(BURST_SIZE, dtype=np.float32)              # [0..15]
    np_in1  = np.arange(BURST_SIZE, dtype=np.float32) * 2.0        # [0..30]
    np_out0 = np.zeros(BURST_SIZE, dtype=np.float32)
    np_out1 = np.zeros(BURST_SIZE, dtype=np.float32)

    sim(np_in0, np_in1, np_out0, np_out1)

    # Each CT performs +1.0 on its data
    np.testing.assert_allclose(np_out0, np_in0 + 1.0, rtol=1e-5)
    np.testing.assert_allclose(np_out1, np_in1 + 1.0, rtol=1e-5)
    print("test_decoupled_2x1_mesh PASSED")
    print(f"  CT0 result (first 4): {np_out0[:4]}")
    print(f"  CT1 result (first 4): {np_out1[:4]}")


if __name__ == "__main__":
    test_decoupled_message_passing()
    test_decoupled_2x1_mesh()

