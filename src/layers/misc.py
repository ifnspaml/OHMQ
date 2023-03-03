import torch
from torch import nn


def flip_x(x: torch.Tensor):
    return x.flip(dims=[3])


def flip_y(x: torch.Tensor):
    return x.flip(dims=[2])


def flip_xy(x: torch.Tensor):
    return x.flip(dims=[2, 3])


class DeterministicReflectancePadding2D(nn.Module):
    """
    Deterministic implementation of torch.nn.ReflectionPad2d
    """
    def __init__(self, padding_size):
        super().__init__()
        if isinstance(padding_size, (tuple, list)) and len(padding_size) == 4:
            self.padding_size = padding_size
        else:
            assert isinstance(padding_size, int), \
                'padding size must be int or tuple of (int, int, int, int)'
            self.padding_size = [padding_size] * 4

    def forward(self, x):
        left, right, top, bottom = self.padding_size
        if left == right == top == bottom == 0:
            return x
        new_shape = [x.shape[0], x.shape[1], x.shape[2] + top + bottom,
                     x.shape[3] + left + right]
        padded = torch.zeros(new_shape, device=x.device, dtype=x.dtype)
        flipped_x = flip_x(x)
        flipped_y = flip_y(x)
        flipped_xy = flip_y(flipped_x)

        right_border = x.shape[3] + left
        bottom_border = x.shape[2] + top
        padded[:, :, top: bottom_border, left: right_border] = x  # x
        # pad in x dimension
        padded[:, :, top:bottom_border, :left] = \
            flipped_x[:, :, :, -(left+1):-1]  # a
        padded[:, :, top:bottom_border, right_border:] = \
            flipped_x[:, :, :, 1: right+1]  # b
        # pad in y dimension
        padded[:, :, :top, left:right_border] = \
            flipped_y[:, :, -(top+1):-1]  # c
        padded[:, :, bottom_border:, left:right_border] = \
            flipped_y[:, :, 1: bottom+1]  # d
        # pad corners
        padded[:, :, :top, :left] = \
            flipped_xy[:, :, -(top+1):-1, -(left+1):-1]  # e
        padded[:, :, :top, right_border:] = \
            flipped_xy[:, :, -(top+1):-1, 1: right+1]  # f
        padded[:, :, bottom_border:, :left] = \
            flipped_xy[:, :, 1: bottom+1, -(left+1): -1]  # g
        padded[:, :, bottom_border:, right_border:] = \
            flipped_xy[:, :, 1: bottom+1, 1: right+1]  # h
        return padded


class ChannelNorm2D(nn.Module):
    """
    Similar to default Torch instanceNorm2D but calculates
    moments over channel dimension instead of spatial dims.
    Expects input_dim in format (B,C,H,W)
    """

    def __init__(self, input_channels, momentum=0.1, eps=1e-3,
                 affine=True, **kwargs):
        super(ChannelNorm2D, self).__init__()

        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x):
        """
        Calculate moments over channel dim, normalize.
        x:  Image tensor, shape (B,C,H,W)
        """
        mu, var = torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1,
                                                                keepdim=True)

        x_normed = (x - mu) * torch.rsqrt(var + self.eps)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta
        return x_normed


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()

        self.activation = nn.ReLU(True)

        pad_size = int((kernel_size-1)/2)
        self.pad = DeterministicReflectancePadding2D(pad_size)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=stride)
        self.norm1 = ChannelNorm2D(channels,
                                   momentum=0.1,
                                   affine=True,
                                   track_running_stats=False)
        self.norm2 = ChannelNorm2D(channels,
                                   momentum=0.1,
                                   affine=True,
                                   track_running_stats=False)

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res)
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)
