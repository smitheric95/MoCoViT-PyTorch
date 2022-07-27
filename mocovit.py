"""Module containing all components necessary for creating a MoCoVit network.

Based on https://arxiv.org/abs/2205.12635v1. Seriously, read the paper!

Variable names try to follow ghostnet.pytorch repository.
"""
import os
import importlib.util
import torch
import torch.nn as nn

# Python hack to import modules containing periods
spec = importlib.util.spec_from_file_location('ghost_net', \
       os.path.join(os.path.dirname(__file__), 'ghostnet.pytorch', 'ghost_net.py'))
ghost_net = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ghost_net)

class MoSA(nn.Module):
    """Mobile Self-Attention Module"""
    def __init__(self, inp: int):
        """Initialize a MoSA module.

        Args:
            inp (int): Input channel size.
        """
        super(MoSA, self).__init__()
        self.dim_head = 64
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dw_conv = ghost_net.depthwise_conv(inp, inp)
        self.ghost_module = ghost_net.GhostModule(inp, inp)

    def forward(self, v: torch.Tensor):
        """Calculate mobile self-attention. See Equation 3.

        Args:
            v (torch.Tensor): Input vector representing the 'Value' in Q, K, V.

        Returns:
            torch.Tensor: Result of mobile self-attention.
        """
        out = torch.matmul(self.attend(torch.matmul(v, v.transpose(-1, -2)) * self.scale), v)
        out = out + self.dw_conv(v)

        return self.ghost_module(out)


class MoFFN(nn.Module):
    """Mobile Feed Forward Network"""
    def __init__(self, inp: int, hidden_dim: int, oup: int):
        """Initialize a MoFNN module.

        Args:
            inp (int): Input channel size.
            hidden_dim (int): Hidden dimension size.
            oup (int): Output channel size.
        """
        super(MoFFN, self).__init__()
        self.ffn = nn.Sequential(
            ghost_net.GhostModule(inp, hidden_dim),
            ghost_net.SELayer(hidden_dim),
            ghost_net.GhostModule(hidden_dim, oup),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass of MoFFN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.ffn(x) + x


class MoTBlock(nn.Module):
    """Mobile Transformer Block"""
    def __init__(self, inp: int, hidden_dim: int, oup: int):
        """Initialize a MoTBlock module.

        Args:
            inp (int): Input channel size.
            hidden_dim (int): Hidden dimension size.
            oup (int): Output channel size.
        """
        super(MoTBlock, self).__init__()
        self.mosa = MoSA(inp)
        self.moffn = MoFFN(inp, hidden_dim, oup)

    def forward(self, x: torch.Tensor):
        """Forward pass of MoTBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        mosa_out = self.mosa(x) + x
        moffn_out = self.moffn(mosa_out)

        return moffn_out + mosa_out


class MoCoViT(nn.Module):
    """Mobile Convolutional Transformer network"""
    def __init__(self, input_channel: int = 160, exp_size: int = 960, num_blocks: int = 4, width_mult: int = 1):
        """Initialize MoCoViT module.

        Args:
            input_channel (int, optional): Number of input channels. Defaults to 160.
            exp_size (int, optional): Exapnsion size. Defaults to 960.
            num_blocks (int, optional): Number of MoTBlocks. Defaults to 4.
            width_mult (int, optional): Width multiplier. Defaults to 1.
        """
        super(MoCoViT, self).__init__()
        self.input_channel = input_channel
        self.exp_size = exp_size
        self.num_blocks = num_blocks
        self.width_mult = width_mult

        self._ghost_net = ghost_net.ghost_net()
        self.ghost_blocks = nn.Sequential(*list(self._ghost_net.children())[0][:13])
        self.squeeze = self._ghost_net.squeeze

        mtb_layers = []
        for _ in range(self.num_blocks):
            output_channel = ghost_net._make_divisible(self.input_channel * self.width_mult, 4)
            hidden_channel = ghost_net._make_divisible(self.exp_size * self.width_mult, 4)
            mtb_layers.append(MoTBlock(self.input_channel, hidden_channel, output_channel))

        self.motblocks = nn.Sequential(*mtb_layers)
        self.classifier = self._ghost_net.classifier


    def forward(self, x: torch.Tensor):
        """Forward pass of MoCoViT.

        Args:
            x (torch.Tensor): Input tensor (batch of images).

        Returns:
            torch.Tensor: Output tensor (classifications).
        """
        x = self.squeeze(self.motblocks(self.ghost_blocks(x)))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
