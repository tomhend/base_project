import torch
import numpy as np
from typing import Tuple


class ConvNetLayer3D(torch.nn.Module):
    """
    A layer used in the ConvNet3d, consists of a convolutional step and optional maxpool and dropout
    steps. Subclasses torch.nn.Module
    """

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
        """
        Initializes the layer.

        Args:
            input_size (list[int]): size of the input
            out_channels (int): number of output channels
            kernel_size (int): size of the convolutional kernel
            stride (int): stride, should be 1 for now (because of using 'same' padding)
            maxpool_size (int, optional): if supplied specifies the maxpool kernel size.
            Defaults to None.
            maxpool_stride (int, optional): if supplied specifies the maxpool stride.
            Defaults to None.
            dropout_chance (float, optional): if supplied specifies the dropout chance.
            Defaults to None.
        """
        super().__init__()

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
        """
        Forward pass through the network

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
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
        """
        Creates the full network consting of the convolutional layers and the classification head.

        Args:
            n_classes (int): number of output classes
            input_size (list[int]): size of the input image
            out_channels_list (list[int]): list that specifies the output channels for each layer
            kernel_sizes (list[int]): list that specifies the kernel sizes for each layer
            strides (list[int]): list that specifies the convolution stride for each layer
            maxpool_sizes (list[int]): list that specifies the maxpool kernel size for each layer
            maxpool_strides (list[int]): list that specifies the maxpool stride for each layer
            dropout_chances (list[float]): list that specifies the dropout chance for each layer
        """
        super().__init__()
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
        flatten = torch.nn.Flatten()
        relu = torch.nn.ReLU()
        classification_head1 = torch.nn.Linear(flat_n, 1024)
        classification_head2 = torch.nn.Linear(1024, 256)
        classification_head3 = torch.nn.Linear(256, n_classes)

        layers.append(flatten)
        layers.append(classification_head1)
        layers.append(relu)
        layers.append(classification_head2)
        layers.append(relu)
        layers.append(classification_head3)

        self.network = torch.nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.network(input)
