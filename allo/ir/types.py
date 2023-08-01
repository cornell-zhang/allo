# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin


class HIRType:
    def __init__(self, bits, name):
        self.bits = bits
        self.name = name

    def __getitem__(self, shape):
        pass

    def __repr__(self):
        return self.name


bool = HIRType(1, "bool")
int1 = HIRType(1, "int1")
int8 = HIRType(8, "int8")
int16 = HIRType(16, "int16")
int32 = HIRType(32, "int32")
int64 = HIRType(64, "int64")
index = HIRType(32, "index")
float16 = HIRType(16, "float16")
float32 = HIRType(32, "float32")
float64 = HIRType(64, "float64")
