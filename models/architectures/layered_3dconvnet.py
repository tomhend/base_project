import torch
import numpy as np
from typing import Tuple


class ConvNetLayer3D(torch.nn.Module):
    def __init__(
        self,
        input_size: list[int],
        out_channels: int,
        kernel_size: int,
        stride: int,
        maxpool_size: int = None,
        maxpool_stride: int = None,
        dropout_chance: float = None,
    ) -> None:
        self.convolution = torch.nn.Conv3d(
            in_channels=input_size[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = (
            torch.nn.MaxPool3d(maxpool_size, maxpool_stride) if maxpool_size else None
        )
        self.dropout = torch.nn.Dropout3d(dropout_chance) if dropout_chance else None

        self.output_size = [out_channels] + [
            ((dim - maxpool_size) // maxpool_stride + 1) for dim in input_size[1:]
        ]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.convolution(input)
        x = self.relu(x)
        if self.maxpool:
            x = self.maxpool(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class ConvNet3D(torch.nn.Module):
    def __init__(
        self,
        n_classes: int,
        input_size: list[int],
        out_channels_list: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        maxpool_sizes: list[int],
        maxpool_strides: list[int],
        dropout_chances: list[float],
    ) -> None:
        layers = []
        for layer_idx in range(len(out_channels_list)):
            layer = ConvNetLayer3D(
                input_size=input_size,
                out_channels=out_channels_list[layer_idx],
                kernel_size=kernel_sizes[layer_idx],
                stride=strides[layer_idx],
                maxpool_size=maxpool_sizes[layer_idx],
                maxpool_stride=maxpool_strides[layer_idx],
                dropout_chance=dropout_chances[layer_idx],
            )
            layers.append(layer)
            input_size = layer.output_size

        flat_n = np.prod(input_size)
        classification_head = torch.nn.Linear(flat_n, n_classes)

        layers.append(classification_head)
        self.network = torch.nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.network(input)
