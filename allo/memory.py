# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from itertools import product


class Layout:
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

    def get_placement(self, mesh_dims):
        """
        Calculate mapping from tensor tile IDs to PE tile IDs based on the placement scheme.

        Args:
            mesh_dims (list): Dimensions of the device mesh (e.g., [4] for 1D, [2,2] for 2D)

        Returns:
            dict: A mapping from tensor tile IDs to corresponding PE tile coordinates
        """
        # Generate all possible PE coordinates
        pe_coords = list(product(*[range(dim) for dim in mesh_dims]))

        # Initialize mapping
        mapping = {}

        # For each PE coordinate, determine its tensor tile ID
        for pe_coord in pe_coords:
            tensor_id_parts = []

            for _, (op, dim) in enumerate(self.placement):
                if op == "S":
                    # For sharding, use the coordinate at the specified dimension
                    # start from right to left
                    mesh_dim = int(dim)
                    tensor_id_parts.append(str(pe_coord[-mesh_dim - 1]))
                elif op == "R":
                    # For replication, use 'R'
                    tensor_id_parts.append("R")

            tensor_id = "".join(tensor_id_parts)

            # Add this PE coordinate to the mapping for this tensor ID
            if tensor_id not in mapping:
                mapping[tensor_id] = []
            mapping[tensor_id].append(pe_coord)

        # Post-process the mapping to combine PE coordinates for replicated dimensions
        result = {}
        for tensor_id, coords in mapping.items():
            # Convert to tuples for final output
            result[tensor_id] = [tuple(coord) for coord in coords]

        return result

    def __repr__(self):
        result = ""
        for letter, number in self.placement:
            result += letter
            if number is not None:
                result += str(number)
        return f"Layout({result})"


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
        for i, s in enumerate(self.shape):
            shard, dim = self.layout.placement[i]
            if shard == "R":
                local_shape.append(s)
            else:
                # count from right to left
                local_shape.append(s // self.mapping[-dim - 1])
        return tuple(local_shape)

    def __str__(self):
        return f"DTensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, layout={self.layout}, mapping={self.mapping}, rank={self.rank}, local_shape={self.get_local_shape()})"
