# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn.functional as F
from typing import Annotated
from allo.ir.types import float32, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

KERNEL_LIB_PATH = "../../../../allo/backend/experimental/kernels/"

Ly = Layout("S1S0")
Ly_1 = Layout("S1")

# Masked Softmax dimensions
SEQ_LEN_TILED = 64
SEQ_LEN = 64  # Sequence length
HEAD_TILE = 3


# PyTorch reference code starts
def masked_softmax_tiled(
    attention_score_tile: Annotated[torch.Tensor, "shape: (16, 64), dtype: float32"],
    tile_row_start: Annotated[int, "shape: (1), dtype: int32"],
) -> Annotated[torch.Tensor, "shape: (16, 64), dtype: float32"]:
    """
    Causal masked softmax for attention scores (tiled version)
    attention_score_tile: raw attention scores tile (TILE_ROWS x SEQ), dtype=float32
    tile_row_start: starting row index of this tile in the full (64, 64) matrix
    Returns: softmax attention weights with causal masking (TILE_ROWS x SEQ), dtype=float32

    Note: This processes a horizontal tile of the attention matrix.
    Each row in the tile gets its own softmax, with causal masking based on global position.
    """
    attention_score_tile = attention_score_tile.view(SEQ_LEN_TILED, HEAD_TILE, SEQ_LEN)
    # Create causal mask for this tile based on global row positions
    mask = torch.zeros(SEQ_LEN_TILED, SEQ_LEN_TILED, dtype=torch.bool)
    for i in range(SEQ_LEN_TILED):
        global_row_idx = tile_row_start + i
        # Mask positions where column_idx > global_row_idx (future tokens)
        mask[i, global_row_idx + 1 :] = True
    mask = mask.unsqueeze(1)
    # Apply mask by setting masked positions to -inf
    masked_scores = attention_score_tile.masked_fill(mask, float("-inf"))
    attn_weights = F.softmax(masked_scores, dim=-1)
    return attn_weights


# PyTorch reference code ends


def _test_masked_softmax_tiled():
    masked_softmax_kernel = ExternalModule(
        top="masked_softmax_float32",
        impl_path=KERNEL_LIB_PATH + "masked_softmax.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    Ty_1 = int32

    @df.region()
    def top():
        @df.kernel(mapping=[2, HEAD_TILE])
        def core(
            Input: Ty[SEQ_LEN_TILED, SEQ_LEN * HEAD_TILE] @ Ly,
            TileRowStart: Ty_1[2] @ Ly_1,
            Output: Ty[SEQ_LEN_TILED, SEQ_LEN * HEAD_TILE] @ Ly,
        ):
            masked_softmax_kernel(Input, TileRowStart, Output)

    # Create random input data
    input_tensor = torch.randn(SEQ_LEN_TILED, SEQ_LEN * HEAD_TILE, dtype=torch.float32)
    tile_row_start = 0
    tile_row_start_tensor = torch.tensor([tile_row_start], dtype=torch.int32)
    output = masked_softmax_tiled(input_tensor, tile_row_start_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie-mlir",
            profile=True,
            warmup=20,
            num_iters=100,  # ! executing only once may get undefined result.
        )
        output_allo = np.zeros((SEQ_LEN_TILED, SEQ_LEN * HEAD_TILE)).astype(np.float32)
        output = output.view(SEQ_LEN_TILED, HEAD_TILE * SEQ_LEN)
        mod(input_tensor.cpu().numpy(), np.array([0, 32]), output_allo)
        np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASS!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_masked_softmax_tiled()
