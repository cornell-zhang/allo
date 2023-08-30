# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32


@pytest.mark.parametrize("axes", [[0], [1], [0, 1]])
def test_systolic_array(axes):
    M, N = 2, 2

    def matrix_add(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N, name="PE"):
            B[i, j] = A[i, j] + 1
        return B

    s = allo.customize(matrix_add)
    print(s.module)
    s.unfold("PE", axes=axes)
    assert str(s.module).count('from = "A"') == len(axes) * M
    assert str(s.module).count('to = "B"') == len(axes) * M
    print(s.module)
    np_A = np.random.randint(0, 10, size=(M, N))
    np_B = np_A + 1
    mod = s.build()
    res_allo = mod(np_A)
    np.testing.assert_allclose(res_allo, np_B)


if __name__ == "__main__":
    pytest.main([__file__])
