# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import tempfile
import pytest
import numpy as np
import scipy.special
import allo.backend.hls as hls

from flash_Atten import get_scheduled_flash_attention


def run_test_with_params(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, BLOCK_T):
    print("=" * 60)
    print(
        f"Testing FlashAttention: BATCH={BATCH_SIZE}, SEQ_LEN={CONTEXT_LENGTH}, "
        f"HIDDEN={HIDDEN_SIZE}, HEADS={NUM_HEADS}, BLOCK_T={BLOCK_T}"
    )
    print("=" * 60)

    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    D_SQRT = HEAD_DIM**0.5
    THREE_H = 3 * HIDDEN_SIZE
    IN_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * THREE_H
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM

    A = np.random.rand(IN_ELEMS).astype(np.float32)
    B_out = np.zeros(OUT_ELEMS, dtype=np.float32)

    A_reshaped = A.reshape((BATCH_SIZE, CONTEXT_LENGTH, 3, NUM_HEADS, HEAD_DIM))
    Q_np = A_reshaped[:, :, 0, :, :].transpose((0, 2, 1, 3))
    K_np = A_reshaped[:, :, 1, :, :].transpose((0, 2, 1, 3))
    V_np = A_reshaped[:, :, 2, :, :].transpose((0, 2, 1, 3))

    scores = np.matmul(Q_np, K_np.transpose((0, 1, 3, 2)))
    scores = scores * (1.0 / D_SQRT)
    attn_weights = scipy.special.softmax(scores, axis=-1)

    out_np = np.matmul(attn_weights, V_np)
    B_golden = out_np.transpose((0, 2, 1, 3)).flatten()

    s = get_scheduled_flash_attention(
        BATCH_SIZE=BATCH_SIZE,
        CONTEXT_LENGTH=CONTEXT_LENGTH,
        HIDDEN_SIZE=HIDDEN_SIZE,
        NUM_HEADS=NUM_HEADS,
        BLOCK_T=BLOCK_T,
    )

    print("Running Software Simulator for numerical correctness...")
    sim_mod = s.build(target="llvm")
    sim_mod(A, B_out)

    try:
        np.testing.assert_allclose(B_out, B_golden, rtol=1e-4, atol=1e-4)
        print("✅ Simulator Test Passed: Outputs match Golden Reference!")
    except AssertionError as e:
        print("❌ Simulator Test Failed!")
        raise e

    if hls.is_available("vitis_hls"):
        print("Running Vitis HLS Synthesis")
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_mod = s.build(target="vitis_hls", mode="csyn", project=tmpdir)
            hls_mod()
            print("✅ HLS Synthesis Passed!")
    else:
        print("⚠️ Vitis HLS not available, skipping C synthesis.")


def test_flashattention():
    run_test_with_params(
        BATCH_SIZE=4, CONTEXT_LENGTH=16, HIDDEN_SIZE=64, NUM_HEADS=4, BLOCK_T=4
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Allo FlashAttention Testbench")

    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=4,
        required=False,
        help="Batch size of input data",
    )
    parser.add_argument(
        "--CONTEXT_LENGTH",
        type=int,
        default=16,
        required=False,
        help="Context length of input data",
    )
    parser.add_argument(
        "--HIDDEN_SIZE",
        type=int,
        default=64,
        required=False,
        help="Hidden size of input data",
    )
    parser.add_argument(
        "--NUM_HEADS",
        type=int,
        default=4,
        required=False,
        help="Number of heads of input data",
    )
    parser.add_argument(
        "--BLOCK_T", type=int, default=4, required=False, help="Size of tiles"
    )

    args = parser.parse_args()

    run_test_with_params(
        BATCH_SIZE=args.BATCH_SIZE,
        CONTEXT_LENGTH=args.CONTEXT_LENGTH,
        HIDDEN_SIZE=args.HIDDEN_SIZE,
        NUM_HEADS=args.NUM_HEADS,
        BLOCK_T=args.BLOCK_T,
    )
