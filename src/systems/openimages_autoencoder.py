from src import oi_models as models
from src.systems import autoencoder_helpers
import torch
from src.layers.quant import QuantLayer, ProbLayer
from src.layers.int_quant import IntQuantLayer
from src.helpers import utils
from src.integer_quant_entropy_model import HyperpriorDensity
from contextlib import contextmanager
from src.helpers import loss as lossfuncs
import pytorch_lightning as pl
import numpy as np


class OpenimagesAutoencoder(pl.LightningModule):
    def __init__(
            self,
            lambda_rd,
            learning_rate,
            weight_decay,
            no_quant,
            integer_quant,
            encoder_channels,
            bottleneck_channels,
            no_quant_symbols,
            *args,
            **kwargs
    ):
        super().__init__()
        self.lambda_rd = lambda_rd
        self.no_quant = no_quant
        self.latent_bits = np.log2(no_quant_symbols) if not no_quant else 32
        self.no_quant_symbols = no_quant_symbols
        self.encoder_channels = encoder_channels
        self.integer_quant = integer_quant
        self.bottleneck_channels = np.array(bottleneck_channels)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.inverse_normalize = utils.inverse_normalization(openimages=True)
        ch = encoder_channels if self.integer_quant else bottleneck_channels[-1]
        self.encoder = models.OpenimagesEncoder(in_channels=3, out_channels=ch)
        self.decoder = models.OpenimagesDecoder(in_ch=ch, out_ch=3)
        if self.integer_quant:
            self.min_symbol, self.max_symbol = -(self.no_quant_symbols // 2 - 1), self.no_quant_symbols // 2
            self.prior_model = HyperpriorDensity(self.encoder_channels, min_s=self.min_symbol, max_s=self.max_symbol)
            self.quant = IntQuantLayer(self.bottleneck_channels, self.no_quant_symbols)
        elif self.ohm_quant:
            self.quant = QuantLayer(
                in_channels=ch,
                no_quant_channels=self.bottleneck_channels,
                no_quant_symbols=self.no_quant_symbols,
            )
            self.prior_model = models.OpenimagesEntropyModel(self.encoder_channels)
            self.prob = ProbLayer(no_quant_channels=self.bottleneck_channels, no_quant_symbols=self.no_quant_symbols)

    def training_step(self, batch, *args, **kwargs):
        x_in, lambda_rd = batch[0], self.lambda_rd[0]
        channel_percentage = torch.sqrt(torch.rand((x_in.shape[0],), device=self.device))
        if len(self.lambda_rd) == 2:
            lambda_rd_min, lambda_rd_max = self.lambda_rd
            lambda_rd += (lambda_rd_max - lambda_rd_min) * channel_percentage
        x_hat, quantized, prior = self(x_in, channel_percentage=channel_percentage)
        loss, rate_loss, distortion_loss = self.calc_loss(x_in, x_hat, quantized, prior, lambda_rd)
        psnr, ssim = autoencoder_helpers.get_image_metrics(self.inverse_normalize, x_in, x_hat)
        self.log_dict(autoencoder_helpers.get_log_dict(
            self.training, channel_percentage, psnr=psnr, rate_loss=rate_loss), prog_bar=True)
        self.log_dict(autoencoder_helpers.get_log_dict(
            self.training, channel_percentage, loss=loss, distortion_loss=distortion_loss))
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x_in, val_loss_sum, channel_percentages = batch[0], 0, autoencoder_helpers.get_val_channel_percentages(self.bottleneck_channels)
        for channel_percentage in channel_percentages:
            x_hat, quantized, prior = self(x_in, channel_percentage)
            psnr, ssim = autoencoder_helpers.get_image_metrics(self.inverse_normalize, x_in, x_hat)
            val_loss, rate_loss, distortion_loss = self.calc_loss(x_in, x_hat, quantized, prior)
            val_loss_sum += val_loss
            self.log_dict(autoencoder_helpers.get_log_dict(
                self.training, channel_percentage,
                psnr=psnr, rate_loss=rate_loss,loss=val_loss, distortion_loss=distortion_loss))
        val_loss = val_loss_sum / len(channel_percentages)
        self.log('val_loss', val_loss)
        if not self.no_quant and not self.integer_quant:
            self.log('temperature', self.quant.temp[0])

    def calc_loss(self, x_in, x_hat, quantized, prior, lambda_rd=None):
        lambda_rd = self.lambda_rd[0] if lambda_rd is None else lambda_rd
        distortion_loss = lossfuncs.calc_distortion_loss(x_in, x_hat)
        if self.no_quant:
            return distortion_loss.mean(), 0, 0, 0
        quantized = torch.ones_like(quantized) if self.integer_quant else quantized
        rate_loss = lossfuncs.calc_rate_loss(x_in, quantized, prior)
        rd_loss = torch.mean(lambda_rd * distortion_loss + (1 - lambda_rd) * rate_loss)
        loss = rd_loss
        return loss.mean(), rate_loss.mean(), distortion_loss.mean()

    def encode(self, x, ch_fraction=1):
        latent = self.encoder(x)
        if self.no_quant:
            return latent, None, torch.ones_like(latent), 0
        elif self.integer_quant:
            return self.encode_int_quant(latent, ch_fraction)
        elif self.ohm_quant:
            return self.encode_ohm_quant(latent, ch_fraction)
        else:
            raise ValueError('Unknown quantization method')

    def encode_int_quant(self, latent, ch_fraction):
        latent_quantized = self.quant(latent)
        latent_prior = self.prior_model(latent_quantized)
        mask = autoencoder_helpers.get_mask(self.bottleneck_channels, ch_fraction).to(latent.device)
        latent_quantized = latent_quantized * mask[:, :, None, None]
        latent_prior = latent_prior * mask[:, :, None, None] + ~mask[:, :, None, None]
        return latent_quantized, latent_quantized, latent_prior

    def encode_ohm_quant(self, latent, ch_fraction):
        latent_dequantized, latent_quantized = self.quant(latent, ch_fraction)
        latent_prior = self.prob(self.prior_model(torch.ones_like(latent[:, :1])))
        return latent_dequantized, latent_quantized, latent_prior

    def forward(self, x, channel_percentage=1):
        temp_padding = utils.TemproaryPadding(down_factor=16, img_height=x.shape[2], img_width=x.shape[3])
        x = temp_padding.apply(x)
        dequantized, quantized, prior = self.encode(x, channel_percentage)
        rec = self.decoder(dequantized)
        rec = temp_padding.unpad(rec)
        return rec, quantized, prior

    @property
    def ohm_quant(self):
        return not (self.no_quant or self.integer_quant)

    @property
    def example_input_array(self):
        return torch.zeros(1, 3, 256, 512, device=self.device)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def on_train_start(self) -> None:
        utils.save_checkpoint_on_sigterm(self.trainer)

