from torch import nn


def mnist_encoder(out_channels):
    out_channels = 32 if out_channels is None else out_channels
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=3, padding=1), nn.ReLU(True),
        nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(True),
        nn.Conv2d(16, out_channels, 3, stride=2, padding=1))


def mnist_decoder(in_channels=32):
    in_channels = 32 if in_channels is None else in_channels
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, 16, 3, stride=2), nn.ReLU(True),
        nn.ConvTranspose2d(16, 8, 5, stride=2, padding=1), nn.ReLU(True),
        nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1), nn.Tanh())


class EntropyModelMnist(nn.Sequential):
    def __init__(self, out_channels):
        super().__init__(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, 1),
        )