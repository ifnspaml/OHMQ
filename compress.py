import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import args as default_args
from torchvision import transforms
from src.systems import MnistAutoencoder, OpenimagesAutoencoder
import torch
from torch import Tensor
import os
import numpy as np
from dataclasses import dataclass
from typing import List
from datetime import datetime
import pandas as pd
from src.helpers import metrics
from src.helpers.torchac_helpers import estimate_bitrate_from_pmf, pmf_to_cdf
from PIL import Image
from src.datasets import FolderDataset
from src.helpers import utils


def calc_num_channels(system, channel_fraction=1):
    if system.no_quant:
        ch_fr = torch.tensor(system.encoder_channels)
    else:
        ch_fr = utils.calc_num_channels(channel_fraction, system.bottleneck_channels[0], system.bottleneck_channels[-1])
    return ch_fr.to(system.device)


@dataclass
class Metrics:
    psnr: float
    ssim: float
    msssim: float
    bitrate_estimated: float
    bitrate_real: float


@dataclass
class Result:
    reconstruction: Tensor
    bytestream: Tensor
    original: Tensor
    metrics: Metrics


def get_ckpt_info(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')  # cpu is fine, since we don't actually load the state dict
    lightning = 'args' not in ckpt
    mnist = ckpt['hyper_parameters']['mnist'] if lightning else ckpt['args'].model == 'mnist'
    return mnist, lightning


def compress_cli():
    args = parse_args()
    compress(
        checkpoint=args.checkpoint,
        image_path=args.image_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        channel_fractions=args.channel_fractions,
        do_save=args.do_save,
        do_plot=args.do_plot
    )


def compress(
        checkpoint,
        image_path,
        batch_size=1,
        output_dir=None,
        channel_fractions=-1,
        do_save=False,
        do_plot=True,
):
    utils.show_tensor_info_in_debugger()
    set_flags()
    ckpt_path = utils.find_ckpt(checkpoint)
    mnist, lightning = get_ckpt_info(ckpt_path)
    output_dir = mk_output_dir(output_dir, checkpoint)
    dataloader = get_dataloader(image_path, batch_size, mnist)
    model_cls = MnistAutoencoder if mnist else OpenimagesAutoencoder
    print('loading checkpoint:', ckpt_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model_cls.load_from_checkpoint(checkpoint).to(device)
    channel_fractions = get_channel_fractions(channel_fractions, model.bottleneck_channels)
    print(model)
    print(f'{model.bottleneck_channels=}')
    results_df = pd.DataFrame()
    for channel_fraction in channel_fractions:
        num_channels = calc_num_channels(model, channel_fraction).item()
        print(f'{channel_fraction=}\t{num_channels=}')
        if num_channels < 1:
            print('num channels must be at least 1')
            continue
        ch_fraction_results = compress_channel_fraction(model, dataloader, mnist, channel_fraction, do_save=do_save,
                                                        output_dir=output_dir)
        save_results(output_dir, ch_fraction_results, channel_fraction)
        avg_results = dict(ch_fraction=channel_fraction, **ch_fraction_results.loc['avg'].to_dict())
        results_df = results_df.append(avg_results, ignore_index=True)

    if len(channel_fractions) > 1 and do_plot:
        x_axis = 'channel_fraction' if model.no_quant else 'bitrate_estimated'
        results_df = results_df.sort_values(x_axis)
        results_df.plot(x=x_axis, y=['psnr', 'ssim'], secondary_y='ssim')
        min_ch, max_ch = model.bottleneck_channels
        title = f'{min_ch} - {max_ch} channels, int_quant: {model.integer_quant}'
        plt.suptitle(title)
        plt.savefig(f'{output_dir}/rate_distortion.png')
        plt.show()
        results_df.to_csv(f'{output_dir}/rate_distortion.csv')
    return results_df


def get_channel_fractions(channel_fractions, bottleneck_channels):
    if len(bottleneck_channels) == 1:
        channel_fractions = [1]
    else:
        if isinstance(channel_fractions, int):
            channel_fractions = [channel_fractions]
        if len(channel_fractions) == 1 and channel_fractions[0] < 0:
            min_channels, max_channels = bottleneck_channels
            num_experiments = max_channels - min_channels + 1
            channel_fractions = list(np.linspace(0, 1, num_experiments))
    return channel_fractions


def set_flags(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_results(output_dir, results: pd.DataFrame, channel_fraction):
    channel_fraction = str(channel_fraction).replace('.', '_')
    metrics_path = os.path.join(output_dir, f'metrics_ch_fraction_{channel_fraction}.csv')
    results.to_csv(metrics_path)
    print('results\n', results.loc['avg'].to_string())


def compress_channel_fraction(model, dataloader, mnist, channel_fraction, quant_metrics=True, do_save=False, output_dir=None) -> List[Metrics]:
    model.eval()
    inv_normalize = utils.inverse_normalization(openimages=not mnist)
    results = []
    for i, img_batch in enumerate(dataloader):
        with torch.no_grad():
            img_batch = img_batch[0].to(model.device)
            n_pixels = img_batch.shape[2] * img_batch.shape[3]
            forward_results = model(img_batch, channel_percentage=channel_fraction)
            reconstruction_batch, quantized_batch, distribution_batch = forward_results[:4]
            for j in range(len(img_batch)):
                img = img_batch[j:j+1]
                reconstruction = reconstruction_batch[j:j+1]
                distribution = distribution_batch[j:j+1]
                # Revert normalization so that the color spaces are normal again
                img = inv_normalize(img).clamp(0, 1)
                reconstruction = inv_normalize(reconstruction).clamp(0, 1)
                image_metrics = calc_image_metrics(img, reconstruction)
                n_bits = model.latent_bits
                n_channels = calc_num_channels(model, channel_fraction).item()
                if not model.no_quant and quant_metrics:
                    rate_estimated, rate_real, byte_stream = calc_bottleneck_metrics(quantized_batch[j:j+1], distribution, n_bits, n_pixels, model, n_channels)
                else:
                    rate_estimated = rate_real = 0
                    byte_stream = b''
                results.append(Metrics(*image_metrics, rate_estimated, rate_real))
                if do_save:
                    with open(os.path.join(output_dir, f'stream_{i}_{j}_ch_fraction_{channel_fraction}.pt'), 'wb') as f:
                        f.write(byte_stream)
                    rec = transforms.ToPILImage()(reconstruction.detach().cpu()[0])
                    rec.save(os.path.join(output_dir, f'reconstruction_{i}_{j}_ch_fraction_{channel_fraction}.png'))
                    rec = transforms.ToPILImage()(img.detach().cpu()[0])
                    rec.save(os.path.join(output_dir, f'original_{i}_{j}_ch_fraction_{channel_fraction}.png'))

    df = pd.DataFrame([r.__dict__ for r in results])
    for col in df.columns:
        df.loc['avg', col] = df[col].mean()
    return df


def calc_bottleneck_metrics(residual, distribution, n_bits, n_pixels, model: OpenimagesAutoencoder, n_channels):
    import torchac  # this takes time, import only if needed
    # this may fail on windows (even though torchac is installed) if 'cl.exe' is not in the path environment variable
    # try adding e.g. C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.33.31629\bin\Hostx86\x86 to path in this case
    s = residual.shape
    if model.integer_quant:
        sym = (residual[:, :n_channels] - model.min_symbol)  # first symbol should be '0'
        distribution = model.prior_model.pdf(residual)
        distribution = distribution[:, :n_channels]
    else:
        residual = residual.view(s[0], int(2**n_bits), -1, s[2], s[3])
        residual = residual[:, :, :n_channels]
        sym = torch.argmax(residual, dim=1)
        distribution = distribution.view(s[0], int(2**n_bits), -1, s[2], s[3])
        distribution = distribution[:, :, :n_channels]
        distribution = distribution.permute(0, 2, 3, 4, 1)
    sym = sym.to(torch.int16).cpu()
    cdf = pmf_to_cdf(distribution).cpu()
    byte_stream = b'' + torchac.encode_float_cdf(cdf, sym, check_input_bounds=True)
    rate_estimated = estimate_bitrate_from_pmf(distribution.cpu(), sym) / n_pixels
    rate_real = len(byte_stream) * 8 / n_pixels
    # sym_reconstructed = torchac.decode_float_cdf(cdf, byte_stream)
    # assert torch.equal(sym, sym_reconstructed), "Could not decode image losslessly"

    return rate_estimated, rate_real, byte_stream


def calc_image_metrics(img, reconstruction):
    psnr = metrics.psnr(img, reconstruction)
    ssim = metrics.ssim(img, reconstruction)
    if img.shape[2] > 28 and img.shape[3] > 28:
        msssim = metrics.msssim(img, reconstruction)
    else:
        msssim = None
    return psnr, ssim, msssim


def parse_args():
    parser = ArgumentParser(description="Training of learnable compression.", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--image_path', type=str, default=default_args.image_path, help='path to single image or directory of images (png and/or jpg) to be compressed')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None, help='path to directory where to save the results')
    parser.add_argument('--do_save', action='store_true', help='save encoded and decoded images (default: only metrics)')
    parser.add_argument('--do_plot', action='store_true', help='plot metrics')
    parser.add_argument('--channel_fractions', nargs='+', type=float, default=[default_args.channel_fraction_inference],
                        help='list of quality levels for adaptive bitrate ')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    return args


def mk_output_dir(output_dir=None, ckpt_path=None):
    if output_dir:
        out_dir = output_dir
    else:
        base_dir = 'inference_results/'
        job_id = utils.get_job_id(base_dir)
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        timestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        out_dir = f'{base_dir}id-{job_id}_{ckpt_name}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_dataloader(image_path, batch_size, mnist):
    transform_list = [
        transforms.ToTensor(),
    ]
    if mnist:
        transform_list.insert(0, transforms.Grayscale())
        transform_list.append(transforms.Normalize(0.5, 0.5))
        transform_list.append(transforms.RandomCrop(28))

    else:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))
    all_transforms = transforms.Compose(transform_list)
    if os.path.isfile(image_path):
        data_loader = [(all_transforms(Image.open(image_path)).unsqueeze(0), None)]
    else:
        dataset = FolderDataset(image_path, transform_list=all_transforms)
        data_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=batch_size)
    return data_loader


if __name__ == '__main__':
    compress_cli()
