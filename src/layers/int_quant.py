import torch
from torch import nn


class IntQuantLayer(nn.Module):
    def __init__(self, channels, no_quant_symbols=2):
        super().__init__()
        self.channels = channels
        self.no_quant_symbols = no_quant_symbols
        self.min_symbol, self.max_symbol = -(self.no_quant_symbols // 2 - 1), self.no_quant_symbols // 2

    def forward(self, x):
        quantized = (x + torch.rand_like(x) - 0.5) if self.training else torch.round(x)
        quantized = quantized.clamp(self.min_symbol, self.max_symbol)
        return quantized
