# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re


class LayoutSpec:
    def __init__(self, placement):
        # R: replicated, S: shared
        # e.g., S0S1R, S0R, RS0
        pattern = r"([A-Z])(\d)?"
        matches = re.findall(pattern, placement)
        result = []
        for letter, number in matches:
            if number:
                result.append((letter, int(number)))
            else:
                result.append((letter, None))
        self.placement = result

    def __repr__(self):
        result = ""
        for letter, number in self.placement:
            result += letter
            if number is not None:
                result += str(number)
        return f"LayoutSpec({result})"


class DTensor:
    """
    Distributed tensor.
    """

    def __init__(self, rank, mapping, shape, dtype, layout, name=None):
        self.rank = rank
        self.mapping = mapping
        self.shape = shape  # global shape
        self.dtype = dtype
        self.layout = layout
        self.name = name

    def get_local_shape(self):
        """
        Get the local shape of the tensor.
        """
        if self.layout is None:
            return self.shape
        local_shape = []
        for i in range(len(self.shape)):
            if self.layout.placement[i][0] == "R":
                local_shape.append(self.shape[i])
            else:
                local_shape.append(self.shape[i] // self.mapping[i])
        return tuple(local_shape)

    def __str__(self):
        return f"DTensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, layout={self.layout}, mapping={self.mapping}, rank={self.rank}, local_shape={self.get_local_shape()})"
