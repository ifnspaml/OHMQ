import numpy as np
import torch
import torch.nn.functional as F
from . import pytorch_msssim


def psnr(real, fake, max_val=1, batch_average=False):
    if batch_average:
        return torch.tensor([psnr(r, f, max_val) for r, f in
                             zip(real.unbind(0), fake.unbind(0))]).mean()
    mse = F.mse_loss(real, fake)
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()


def psnr_in_roi(real, fake, mask, max_val=1):
    if mask is None:
        out = None
    else:
        summed_squared_erros = ((real - fake) ** 2 * mask).sum()
        mse_in_roi = summed_squared_erros / (mask.sum() * 3 + 1e-5)
        out = 20 * torch.log10(max_val / torch.sqrt(mse_in_roi))
        out = out.item()
    return out


def ssim(real, fake, window_size=11):
    return pytorch_msssim.ssim(real, fake, window_size).item()


def msssim(real, fake, window_size=11):
    try:
        return pytorch_msssim.msssim(real, fake, window_size).item()
    except RuntimeError:
        return None


def snr(real, fake):
    signal = torch.var(real, unbiased=False)
    noise = F.mse_loss(fake, real)
    return 10 * torch.log10(signal / noise)
