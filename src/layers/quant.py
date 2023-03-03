import torch
from torch import nn
import numpy as np
from src.helpers.utils import ste, calc_num_channels
from einops import rearrange


class QuantLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            no_quant_channels,
            no_quant_symbols=2,
            temp=100.0,
    ):
        super().__init__()
        self.no_channel_symbols = no_quant_symbols
        self.no_channels_min, self.no_channels = np.array(no_quant_channels)[[0, -1]]
        self.softmax = nn.Softmax(dim=1)
        self.softsign = torch.nn.Softsign()
        self.temp = nn.Parameter(torch.ones(self.no_channels) * temp, requires_grad=True)
        self.conv_before = nn.Sequential(nn.ReLU(), nn.Conv2d(in_channels, self.no_channels*self.no_channel_symbols, 1))
        self.conf_after = nn.Sequential(nn.Conv2d(self.no_channels*self.no_channel_symbols, in_channels, 1), nn.ReLU())

    def forward(self, x, channel_percentage=1):
        """
        Parameters
        ----------
        x: tensor with dimensions: batch, (symbols, channels), *other_dims
        channel_percentage  # either a scalar or a tensor of size (batch_size)
        Returns
        -------
        """
        x = self.conv_before(x)
        original_shape = x.shape
        x = -torch.abs(x)
        x = rearrange(x, 'b (s c) h w -> b s c (h w)', s=self.no_channel_symbols, c=self.no_channels)
        temp = self.temp
        softmax = self.softmax(x * temp[None, None, :, None])
        max_idx = torch.argmax(x, dim=1, keepdim=True)
        one_hot = torch.zeros_like(x).scatter_(1, max_idx, 1)
        quantized = ste(one_hot, backward_val=softmax)
        n_channels = calc_num_channels(channel_percentage, self.no_channels_min, self.no_channels).to(x.device)
        mask = torch.arange(self.no_channels, device=x.device).repeat(n_channels.shape[0], 1) < n_channels[:, None]
        quantized = quantized * mask[:, None, :, None]
        quantized = quantized.reshape(original_shape)  # easiest since einsum cannot infer sizes (b, h, w)
        dequantized = self.conf_after(quantized)
        return dequantized, quantized


class ProbLayer(nn.Module):
    def __init__(self, no_quant_channels, no_quant_symbols=2):
        super().__init__()
        self.no_channel_symbols = no_quant_symbols
        self.no_channels = no_quant_channels[-1]
        self.softmax = nn.Softmax(dim=1)
        self.temp = nn.Parameter(torch.ones((self.no_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = -torch.abs(x)
        x = rearrange(x, 'b (s c) h w -> b s c h w', s=self.no_channel_symbols, c=self.no_channels)
        softmax = self.softmax(x * self.temp)
        softmax = rearrange(softmax, 'b s c h w -> b (s c) h w')
        return softmax
