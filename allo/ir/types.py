# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin


class AlloType:
    def __init__(self, bits, name):
        self.bits = bits
        self.name = name

    def __getitem__(self, shape):
        pass

    def __repr__(self):
        return self.name


bool = AlloType(1, "bool")
int1 = AlloType(1, "int1")
int8 = AlloType(8, "int8")
int16 = AlloType(16, "int16")
int32 = AlloType(32, "int32")
int64 = AlloType(64, "int64")
index = AlloType(32, "index")
float16 = AlloType(16, "float16")
float32 = AlloType(32, "float32")
float64 = AlloType(64, "float64")
