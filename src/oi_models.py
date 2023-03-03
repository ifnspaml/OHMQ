from torch import nn
from src.layers.misc import DeterministicReflectancePadding2D, ChannelNorm2D, ResidualBlock

channel_norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)


class OpenimagesEncoder(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv_kwargs = dict(kernel_size=3, stride=2)
        super().__init__(
            DeterministicReflectancePadding2D(3),
            nn.Conv2d(in_channels, 60, kernel_size=(7, 7), stride=1),
            ChannelNorm2D(60, **channel_norm_kwargs),
            nn.ReLU(True),
            DeterministicReflectancePadding2D((0, 1, 1, 0)),
            nn.Conv2d(60, 120, **conv_kwargs),
            ChannelNorm2D(120, **channel_norm_kwargs),
            nn.ReLU(True),
            DeterministicReflectancePadding2D((0, 1, 1, 0)),
            nn.Conv2d(120, 240, **conv_kwargs),
            ChannelNorm2D(240, **channel_norm_kwargs),
            nn.ReLU(True),
            DeterministicReflectancePadding2D((0, 1, 1, 0)),
            nn.Conv2d(240, 480, **conv_kwargs),
            ChannelNorm2D(480, **channel_norm_kwargs),
            nn.ReLU(True),
            DeterministicReflectancePadding2D((0, 1, 1, 0)),
            nn.Conv2d(480, out_channels, **conv_kwargs),
            ChannelNorm2D(out_channels, **channel_norm_kwargs),
        )


def dec_upsample(in_features=960, out_features=3):
        conv_t_kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, 480, **conv_t_kwargs),
            ChannelNorm2D(480, **channel_norm_kwargs),
            nn.ReLU(),
            nn.ConvTranspose2d(480, 240, **conv_t_kwargs),
            ChannelNorm2D(240, **channel_norm_kwargs),
            nn.ReLU(),
            nn.ConvTranspose2d(240, 120, **conv_t_kwargs),
            ChannelNorm2D(120, **channel_norm_kwargs),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 60, **conv_t_kwargs),
            ChannelNorm2D(60, **channel_norm_kwargs),
            nn.ReLU(),
            DeterministicReflectancePadding2D(3),
            nn.Conv2d(60, out_features, kernel_size=(7, 7), stride=1),
        )


class OpenimagesDecoder(nn.Module):
    def __init__(self, in_ch=960, out_ch=3):
        super().__init__()
        self.blocks = nn.Sequential(*([ResidualBlock(channels=in_ch) for _ in range(8)]))
        self.upsampling = dec_upsample(in_features=in_ch, out_features=out_ch)

    def forward(self, x):
        x = x + self.blocks(x)
        x_hat = self.upsampling(x)
        return x_hat


class OpenimagesEntropyModel(nn.Sequential):
    """
    predicts latent from static input
    used in:
     - openimages
    """
    def __init__(self, out_channels):
        super().__init__(
            DeterministicReflectancePadding2D(1),
            nn.Conv2d(1, 480, 3),
            ChannelNorm2D(480, **channel_norm_kwargs),
            nn.ReLU(),
            DeterministicReflectancePadding2D(1),
            nn.Conv2d(480, 960, 3),
            ChannelNorm2D(960, **channel_norm_kwargs),
            nn.ReLU(),
            nn.Conv2d(960, out_channels, kernel_size=1),
        )