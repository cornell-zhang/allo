# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from itertools import product


class Layout:
    """
      Example:

      mesh_dim = [2, 2, 2]
        +-----+
     2 /|    /|
      +-+---+ +
    1 |/    |/
      +-----+
         0
      2D tensor: [32, 32]
      +-----------+
      | 0,0 | 0,1 |
      +-----------+
      | 1,0 | 1,1 |
      +-----------+

      placement = "S2S0" ->   (tensor_dim[-1], shard on mesh_dim[2]),
                              (tensor_dim[-2], shard on mesh_dim[0])

      PE tile (a, ?, b) gets tensor tile (a, b)
    """

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
        self.mapping = mapping  # mesh dims
        self.shape = shape  # tensor shape
        self.dtype = dtype
        self.layout = layout
        self.name = name
        if layout is not None and mapping is not None:
            self.global_placement: dict[str, tuple] = layout.get_placement(mapping)
        self.type_as_param: list = None

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

    def get_access_pattern(self) -> tuple[list, list, list]:
        """
        Specify how to access the dtensor (local tensor) from the global tensor
            (tensor has at most 4 dimensions: DMA support 4-dimension address generation)

        Returns:
            - device_dims (list): Indexes of tensor dimensions sharded across devices.
            - size (list): 4D tensor dimensions used for access.
            - stride (list): Stride along each dimension in the global tensor.
        """
        partition_str = "".join([p[0] for p in self.layout.placement])
        partition_dim = [p[1] for p in self.layout.placement]
        if len(self.shape) == 1:
            if partition_str == "S":
                shard_size = self.shape[0] // self.mapping[-partition_dim[0] - 1]
                device_dims = [2]  # partition idx = 2
                size = [1, 1, self.mapping[-partition_dim[0] - 1], shard_size]
                stride = [0, 0, shard_size, 1]
            elif partition_str == "R":
                device_dims = []  # no partition
                size = [1, 1, 1, self.shape[0]]
                stride = [0, 0, 0, 1]
            else:
                raise ValueError("Unsupported access pattern for 1D tensor.")
        elif len(self.shape) == 2:
            tensor_m, tensor_n = self.shape  # [tensor_m x tensor_n]
            if partition_str == "SS":
                device_a, device_b = (
                    self.mapping[-partition_dim[0] - 1],
                    self.mapping[-partition_dim[1] - 1],
                )
                device_dims = [0, 1]
                size = [device_a, device_b, tensor_m // device_a, tensor_n // device_b]
                stride = [
                    (tensor_m // device_a) * tensor_n,
                    tensor_n // device_b,
                    tensor_n,
                    1,
                ]
            elif partition_str == "SR":
                device_a = self.mapping[-partition_dim[0] - 1]
                # First dim sharded across all devices, second replicated
                device_dims = [1]
                size = [1, device_a, tensor_m // device_a, tensor_n]
                stride = [0, (tensor_m // device_a) * tensor_n, tensor_n, 1]
            elif partition_str == "RS":
                device_b = self.mapping[-partition_dim[1] - 1]
                # First dim replicated, second sharded across second dim of mesh
                device_dims = [1]
                size = [1, device_b, tensor_m, tensor_n // device_b]
                stride = [0, tensor_n // device_b, tensor_n, 1]
            elif partition_str == "RR":
                # Both dimensions replicated
                device_dims = []
                size = [1, 1, tensor_m, tensor_n]
                stride = [0, 0, tensor_n, 1]
            else:
                raise ValueError("Unsupported access pattern for 2D tensor.")
        else:
            raise ValueError("Unsupported access pattern.")

        return device_dims, size, stride

    def __str__(self):
        return f"DTensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, layout={self.layout}, mapping={self.mapping}, rank={self.rank}, local_shape={self.get_local_shape()})"
