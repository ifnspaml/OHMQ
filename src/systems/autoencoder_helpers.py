from config import args as default_args
import torch
from src.helpers import metrics


def get_val_channel_percentages(bottleneck_channels):
    if len(bottleneck_channels) == 2:
        channel_percentages_list = [0.33, 0.66, 1.0]
    else:
        channel_percentages_list = [1.0]
    return channel_percentages_list


def get_mask(bottleneck_channels, ch_fraction):
    if isinstance(ch_fraction, (int, float)):
        ch_fraction = torch.tensor([ch_fraction], dtype=torch.float32)
    min_ch, ch = bottleneck_channels[[0, -1]]
    n_channels = min_ch + torch.ceil(ch_fraction * (ch - min_ch)).int()
    mask = torch.arange(ch, device=n_channels.device)[None, :] < n_channels[:, None]
    return mask

def get_log_dict(is_training, channel_fraction, **kwargs):
    prefix = 'train_' if is_training else 'val_'
    postfix = '' if is_training else f'_{channel_fraction}'
    return {prefix + k + postfix: v for k, v in kwargs.items()}


def add_general_autoencoder_args(parent_parser):
    parser = parent_parser.add_argument_group("BaseAutoencoder")
    parser.add_argument('-bottleneck_channels', nargs='+', type=int, default=default_args.bottleneck_channels)
    parser.add_argument('-encoder_channels', type=int, default=default_args.encoder_channels)
    parser.add_argument('-no_quant_symbols', type=int, default=default_args.no_quant_symbols)
    parser.add_argument('-no_quant', action="store_true")
    parser.add_argument('-lambda_rd', nargs='+', type=float, default=default_args.lambda_rd)
    parser.add_argument('-learning_rate', type=float, default=default_args.learning_rate, help="Adam learning rate")
    parser.add_argument('-weight_decay', type=float, default=default_args.weight_decay, help="Weight decay")
    parser.add_argument('-integer_quant', action='store_true')
    return parent_parser


def get_image_metrics(inverse_normalization, x_in, x_hat):
    with torch.no_grad():
        x_in, x_hat = inverse_normalization(x_in), inverse_normalization(x_hat)
        psnr = metrics.psnr(x_in, x_hat)
        ssim = metrics.ssim(x_in, x_hat)
        return psnr, ssim