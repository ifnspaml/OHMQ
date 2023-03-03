import torch
from einops import rearrange


def calc_rate_loss(x_in, unquantized, prior, per_batch=True) -> torch.Tensor:
    eps = 1e-8
    if per_batch:
        n_pixels = x_in[:1, :1].numel()
        reduce_channels = (1, 2, 3)
    else:
        n_pixels = x_in[:, :1].numel()
        reduce_channels = (1, 2, 3, 4)
    rate_loss_per_channel = torch.sum(-(unquantized * torch.log2(prior + eps)) / n_pixels, dim=reduce_channels)
    return rate_loss_per_channel


def calc_distortion_loss(x_in, x_hat, per_batch=True):
    reduce_channels = (1, 2, 3) if per_batch else (1, 2, 3, 4)
    return torch.nn.MSELoss(reduction='none')(x_in, x_hat).mean(reduce_channels)
