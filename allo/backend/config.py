# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

DEFAULT_CONFIG = {
    "device": "u280",
    "frequency": 300,
    "mode": "csim|csynth|cosim|impl",
}

PART_NUMBER = {
    # Reference: https://github.com/Xilinx/XilinxBoardStore/tree/2022.2
    # Embedded
    "ultra96v2": "xczu3eg-sbva484-1-i",
    "pynqz2": "xc7z020clg400-1",
    "zedboard": "xc7z020clg484-1",
    # Zynq
    "zcu102": "xczu9eg-ffvb1156-2-e",
    "zcu104": "xczu7ev-ffvc1156-2-e",
    "zcu106": "xczu7ev-ffvc1156-2-e",
    "zcu111": "xczu28dr-ffvg1517-2-e",
    # Versal
    "vck190": "xcvc1902-vsva2197-2MP-e-S",
    "vhk158": "xcvh1582-vsva3697-2MP-e-S-es1",
    # Alveo
    # https://github.com/Xilinx/XilinxBoardStore/pull/434
    "u200": "xcu200-fsgd2104-2-e",
    "u250": "xcu250-figd2104-2L-e",
    "u280": "xcu280-fsvh2892-2L-e",
}
