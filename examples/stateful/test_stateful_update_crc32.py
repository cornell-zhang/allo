# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import allo
from allo.ir.types import uint8, uint32, stateful


def update_crc32(data: uint8) -> uint32:
    crc: stateful(uint32) = 0xFFFFFFFF
    crc = crc ^ data
    for j in range(8):
        crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)))
    return crc ^ 0xFFFFFFFF


s = allo.customize(update_crc32)
print(s.module)
code = s.build(target="vhls")
print(code)
