# pylint: disable=too-many-instance-attributes, redundant-returns-doc, unsupported-binary-operation
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from itertools import product
from dataclasses import dataclass


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
        ! Unsafe!! (12,1) is same as (1,21)
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

    def get_placement_exp(self, mesh_dims):
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
                    tensor_id_parts.append(int(pe_coord[-mesh_dim - 1]))
                elif op == "R":
                    # For replication, use 'R'
                    tensor_id_parts.append("R")

            tensor_id = tuple(tensor_id_parts)

            # Add this PE coordinate to the mapping for this tensor ID
            if tensor_id not in mapping:
                mapping[tensor_id] = []
            mapping[tensor_id].append(pe_coord)

        # Post-process the mapping to combine PE coordinates for replicated dimensions
        result: dict[tuple[int | str, ...], list[tuple[int, ...]]] = {}
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
        self.layout: Layout = layout
        self.name = name
        if layout is not None and mapping is not None:
            # tensor tile ID -> PE tile IDs
            self.global_placement: dict[
                tuple[int | str, ...], list[tuple[int, ...]]
            ] = layout.get_placement_exp(mapping)
        self.access_pattern_set = False
        self.global_id = None
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

    def set_global_id(self, global_id: int):
        self.global_id = global_id

    def set_access_pattern(self):
        """
        Specify how to access the dtensor (local tensor) from the global tensor
            (tensor has at most 4 dimensions: DMA support 4-dimension address generation)
        Set offset map for each tensor tile.

        Returns:
            - device_dims (list): Indexes of tensor dimensions sharded across devices.
            - size (list): 4D tensor dimensions used for access.
            - stride (list): Stride along each dimension in the global tensor.
        """
        if self.access_pattern_set:
            return
        self.access_pattern_set = True
        # tensor tile ID -> address offset
        self.offset_map: dict[tuple[int | str, ...], Offset4D] = {}
        partition_str = "".join([p[0] for p in self.layout.placement])
        partition_dim = [p[1] for p in self.layout.placement]
        if len(self.shape) == 1:
            if partition_str == "S":
                dim = partition_dim[0]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(0, 0, i, 0)
                shard_size = self.shape[0] // self.mapping[-dim - 1]
                device_dims = [2]  # partition idx = 2
                size = [1, 1, self.mapping[-dim - 1], shard_size]
                stride = [0, 0, shard_size, 1]
            elif partition_str == "R":
                for key in self.global_placement.keys():
                    self.offset_map[key] = Offset4D(0, 0, 0, 0)
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
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_b, i % device_b, 0, 0)
                device_dims = [0, 1]
                size = [device_a, device_b, tensor_m // device_a, tensor_n // device_b]
                stride = [
                    (tensor_m // device_a) * tensor_n,
                    tensor_n // device_b,
                    tensor_n,
                    1,
                ]
            elif partition_str == "SR":  # TODO: something is wrong here
                device_a = self.mapping[-partition_dim[0] - 1]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_a, i % device_a, 0, 0)
                # First dim sharded across all devices, second replicated
                device_dims = [1]
                size = [1, device_a, tensor_m // device_a, tensor_n]
                stride = [0, (tensor_m // device_a) * tensor_n, tensor_n, 1]
            elif partition_str == "RS":
                device_b = self.mapping[-partition_dim[1] - 1]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_b, i % device_b, 0, 0)
                # First dim replicated, second sharded across second dim of mesh
                device_dims = [1]
                size = [1, device_b, tensor_m, tensor_n // device_b]
                stride = [0, tensor_n // device_b, tensor_n, 1]
            elif partition_str == "RR":
                for key in self.global_placement.keys():
                    self.offset_map[key] = Offset4D(0, 0, 0, 0)
                # Both dimensions replicated
                device_dims = []
                size = [1, 1, tensor_m, tensor_n]
                stride = [0, 0, tensor_n, 1]
            else:
                raise ValueError("Unsupported access pattern for 2D tensor.")
        else:
            raise ValueError("Unsupported access pattern.")
        self.shared_dims, self.size, self.stride = device_dims, size, stride

    def PE_tile_id_to_tensor_tile_id(
        self, pe_tile_id: tuple[int, ...]
    ) -> tuple[int | str, ...]:
        for tensor_tile_id, pe_tile_ids in self.global_placement.items():
            if pe_tile_id in pe_tile_ids:
                return tensor_tile_id
        raise ValueError(
            f"PE tile ID {pe_tile_id} not found in {self.global_placement}"
        )

    def __str__(self):
        return f"DTensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, layout={self.layout}, mapping={self.mapping}, rank={self.rank}, local_shape={self.get_local_shape()})"

    def __repr__(self):
        return f"{self.name}"


# ############################################################
# 4D Addressing
# ############################################################
@dataclass(frozen=True)
class Offset4D:
    """
    4D offset.
    indexed from left to right.
        offset_a: offset along mesh_dim[0]
        offset_b: offset along mesh_dim[1]
        offset_c: offset along mesh_dim[2]
        offset_d: offset along mesh_dim[3]
    """

    offset_a: int
    offset_b: int
    offset_c: int
    offset_d: int

    def get_offset(self, dim: int) -> int:
        if dim == 0:
            return self.offset_a
        if dim == 1:
            return self.offset_b
        if dim == 2:
            return self.offset_c
        if dim == 3:
            return self.offset_d
        raise ValueError(f"Invalid dimension: {dim}")

    def get_next_offset(self, dim: int) -> "Offset4D":
        if dim == 0:
            return Offset4D(
                self.offset_a + 1, self.offset_b, self.offset_c, self.offset_d
            )
        if dim == 1:
            return Offset4D(
                self.offset_a, self.offset_b + 1, self.offset_c, self.offset_d
            )
        if dim == 2:
            return Offset4D(
                self.offset_a, self.offset_b, self.offset_c + 1, self.offset_d
            )
        if dim == 3:
            return Offset4D(
                self.offset_a, self.offset_b, self.offset_c, self.offset_d + 1
            )
        raise ValueError(f"Invalid dimension: {dim}")

    def check_next_offset(self, next_: "Offset4D") -> bool:
        """
        Check whether next_ is the next offset of self
        """
        diffs = [
            next_.offset_a - self.offset_a,
            next_.offset_b - self.offset_b,
            next_.offset_c - self.offset_c,
            next_.offset_d - self.offset_d,
        ]
        if diffs.count(0) == 3 and diffs.count(1) == 1:
            indices = [i for i, diff in enumerate(diffs) if diff == 1]
            return indices[0]
        return -1

    def to_list(self) -> list[int]:
        return [self.offset_a, self.offset_b, self.offset_c, self.offset_d]

    def __eq__(self, other) -> bool:
        return (
            self.offset_a == other.offset_a
            and self.offset_b == other.offset_b
            and self.offset_c == other.offset_c
            and self.offset_d == other.offset_d
        )

    def __hash__(self) -> int:
        return hash((self.offset_a, self.offset_b, self.offset_c, self.offset_d))

    def __str__(self) -> str:
        return f"offset4D ({self.offset_a}, {self.offset_b}, {self.offset_c}, {self.offset_d})"

    def __repr__(self) -> str:
        return self.__str__()


class Size4D:
    """
    4D size.
    indexed from left to right.
        size_a: size along mesh_dim[0]
        size_b: size along mesh_dim[1]
        size_c: size along mesh_dim[2]
        size_d: size along mesh_dim[3]
    """

    def __init__(self, size_a: int, size_b: int, size_c: int, size_d: int):
        self.size_a = size_a
        self.size_b = size_b
        self.size_c = size_c
        self.size_d = size_d

    def copy(self) -> "Size4D":
        return Size4D(self.size_a, self.size_b, self.size_c, self.size_d)

    def get_k_slice(self, k: int) -> "Size4D":
        """
        get a slice of size k
        """
        size_list = [self.size_a, self.size_b, self.size_c, self.size_d]
        dim = 3
        while dim >= 0 and k >= size_list[dim]:
            assert k % size_list[dim] == 0, "Invalid slice size"
            k //= size_list[dim]
            dim -= 1
        size_list[dim] = k
        dim -= 1
        while dim >= 0:
            size_list[dim] = 1
            dim -= 1
        return Size4D.from_list(size_list)

    @classmethod
    def from_list(cls, size_list: list[int]) -> "Size4D":
        if len(size_list) > 4:
            raise ValueError(
                f"Size4D must have at most 4 dimensions, but got {len(size_list)}"
            )
        while len(size_list) < 4:
            size_list.insert(0, 1)
        return cls(*size_list)

    @staticmethod
    def coalesce(size_1: "Size4D", size_2: "Size4D") -> "Size4D":
        return Size4D(
            (size_1.size_a * size_2.size_a),
            (size_1.size_b * size_2.size_b),
            (size_1.size_c * size_2.size_c),
            (size_1.size_d * size_2.size_d),
        )

    @staticmethod
    def subtract(a: "Size4D", b: "Size4D") -> "Size4D":
        list_a, list_b = a.to_list(), b.to_list()
        sub = False
        for i in range(4):
            if list_a[i] != list_b[i]:
                if sub:
                    raise ValueError("Cannot subtract")
                sub = True
                list_a[i] = list_a[i] - list_b[i]
        if not sub:
            return Size4D.from_list([0, 0, 0, 0])
        return Size4D.from_list(list_a)

    @staticmethod
    def divide(a: "Size4D", b: "Size4D") -> "Size4D":
        list_a, list_b = a.to_list(), b.to_list()
        for i in range(4):
            assert list_a[i] % list_b[i] == 0, "invalid division"
            list_a[i] //= list_b[i]
        return Size4D.from_list(list_a)

    @staticmethod
    def multiply(a: "Size4D", b: "Size4D") -> "Size4D":
        list_a, list_b = a.to_list(), b.to_list()
        for i in range(4):
            list_a[i] *= list_b[i]
        return Size4D.from_list(list_a)

    def get_dim_size(self, dim: int) -> int:
        if dim == 0:
            return self.size_a
        if dim == 1:
            return self.size_b
        if dim == 2:
            return self.size_c
        if dim == 3:
            return self.size_d
        raise ValueError(f"Invalid dimension: {dim}")

    def set_dim_size(self, dim: int, size: int):
        if dim == 0:
            self.size_a = size
        elif dim == 1:
            self.size_b = size
        elif dim == 2:
            self.size_c = size
        elif dim == 3:
            self.size_d = size
        else:
            raise ValueError(f"Invalid dimension: {dim}")

    def inc_on_dim(self, dim: int):
        if dim == 0:
            self.size_a += 1
        elif dim == 1:
            self.size_b += 1
        elif dim == 2:
            self.size_c += 1
        elif dim == 3:
            self.size_d += 1
        else:
            raise ValueError(f"Invalid dimension: {dim}")

    def get_total_size(self) -> int:
        return self.size_a * self.size_b * self.size_c * self.size_d

    def to_list(self) -> list[int]:
        return [self.size_a, self.size_b, self.size_c, self.size_d]

    def __eq__(self, other) -> bool:
        return (
            self.size_a == other.size_a
            and self.size_b == other.size_b
            and self.size_c == other.size_c
            and self.size_d == other.size_d
        )

    def __hash__(self) -> int:
        return hash((self.size_a, self.size_b, self.size_c, self.size_d))

    def __str__(self) -> str:
        return f"size4D ({self.size_a}, {self.size_b}, {self.size_c}, {self.size_d})"

    def __repr__(self) -> str:
        return self.__str__()


def coalesce_memory_access(offset_map: dict[Offset4D, list]):
    """
    Coalesce memory tile access.
        The default way is sending each tiling separately.
        But we can try to coalesce some.
    """
    offsets = list(offset_map.keys())
    access: dict[Offset4D, Size4D] = {offset: Size4D(1, 1, 1, 1) for offset in offsets}
    coalesce_info: dict[Offset4D, list[Offset4D]] = {
        offset: [offset] for offset in offsets
    }
    connected_nodes: dict[Offset4D, list[list]] = {
        offset: [offset_map[offset]] for offset in offsets
    }
    coalesce_dim = 3
    while coalesce_dim >= 0:
        sorted_offsets = sorted(
            access.keys(),
            key=lambda x: (x.offset_a, x.offset_b, x.offset_c, x.offset_d),
        )
        coalesed = set()
        base_offset, inc_offset, base_size = None, None, None
        for offset in sorted_offsets:
            if offset in coalesed:
                continue
            if base_offset is None:
                base_offset, inc_offset, base_size = offset, offset, access[offset]
            else:
                inc_offset = inc_offset.get_next_offset(coalesce_dim)
                if inc_offset in access:
                    base_size.inc_on_dim(coalesce_dim)
                    coalesed.add(offset)
                    coalesce_info[base_offset].extend(coalesce_info[inc_offset])
                    connected_nodes[base_offset].extend(connected_nodes[inc_offset])
                else:
                    base_offset, inc_offset, base_size = offset, offset, access[offset]
        for offset in coalesed:
            access.pop(offset)
            coalesce_info.pop(offset)
            connected_nodes.pop(offset)
        coalesce_dim -= 1
    return access, coalesce_info, connected_nodes
