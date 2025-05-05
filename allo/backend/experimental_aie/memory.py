# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ...memory import Layout, DTensor


class AIE_DTensor:
    """
    Distributed tensor used in experimental AIE
    """

    def __init__(self, dtensor: DTensor):
        self.rank = dtensor.rank  # dtensor idx
        self.mapping = dtensor.mapping  # global mapping
        self.shape = dtensor.shape  # global shape
        self.dtype = dtensor.dtype
        self.layout: Layout = dtensor.layout
        self.name = dtensor.name
        self.global_placement: dict[str, tuple] = dtensor.layout.get_placement(
            dtensor.mapping
        )
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
            - device_dims: make partition at which dimensions
            - size: tensor size
            - stride: access stride in global tensor
        """
        partition_str = "".join([p[0] for p in self.layout.placement])
        if len(self.shape) == 1:
            if partition_str == "S":
                shard_size = self.shape[0] // self.mapping[0]
                device_dims = [2]  # partition idx = 2
                size = [1, 1, self.mapping[0], shard_size]
                stride = [0, 0, shard_size, 1]
            elif partition_str == "R":
                device_dims = []  # no partition
                size = [1, 1, 1, self.shape[0]]
                stride = [0, 0, 0, 1]
            else:
                raise ValueError("Unsupported access pattern for 1D tensor.")
        elif len(self.shape) == 2:
            tensor_m, tensor_n = self.shape  # [tensor_m x tensor_n]
            device_a, device_b = None, None  # 2D device to be mapped
            if len(self.mapping) == 1:
                device_a, device_b = 1, self.mapping[0]
            elif len(self.mapping) == 2:
                partition = self.layout.placement
                if partition[0][0] == "S":
                    partition[1] = (partition[1][0], 1 - partition[0][1])
                elif partition[1][0] == "S":  # partition[0][0] == "R"
                    partition[0] = (partition[0][0], 1 - partition[1][1])
                else:
                    partition[0] = (partition[0], 1)
                    partition[1] = (partition[1], 0)
                device_a, device_b = (
                    self.mapping[-partition[0][1] - 1],
                    self.mapping[-partition[1][1] - 1],
                )
            else:
                device_a, device_b = (
                    self.mapping[-partition[0][1] - 1],
                    self.mapping[-partition[1][1] - 1],
                )
            if partition_str == "SS":
                device_dims = [0, 1]
                size = [device_a, device_b, tensor_m // device_a, tensor_n // device_b]
                stride = [
                    (tensor_m // device_a) * tensor_n,
                    tensor_n // device_b,
                    tensor_n,
                    1,
                ]
            elif partition_str == "SR":
                # First dim sharded across all devices, second replicated
                total_devices = device_a * device_b
                device_dims = [1]
                size = [1, total_devices, tensor_m // total_devices, tensor_n]
                stride = [0, (tensor_m // total_devices) * tensor_n, tensor_n, 1]
            elif partition_str == "RS":
                # First dim replicated, second sharded across second dim of mesh
                device_dims = [1]
                size = [1, device_b, tensor_m, tensor_n // device_b]
                stride = [
                    (tensor_m * tensor_n) // (device_a * device_b),
                    tensor_n // device_b,
                    tensor_n,
                    1,
                ]
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
        return f"AIE DTensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, layout={self.layout}, mapping={self.mapping}, rank={self.rank}, local_shape={self.get_local_shape()})"
