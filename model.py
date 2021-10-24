import torch
import torch.nn as nn
import numpy as np
from helpers import encode, decode, decode_torch

class SpaceshipDetector(nn.Module):
    def __init__(self):
        super(SpaceshipDetector, self).__init__()

        self.featureExtractor = FeatureExtractor()
        self.localizingHead   = LocalizingHead()

    def forward(self, x):
        # use CNN (Feature Extractor) to extract higher level features from the image
        features = self.featureExtractor(x)
        # use FC layers (Localizing Head) to compute bbox identifiers
        return self.localizingHead(features)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.layer0 = conv_block(in_channel=1, out_channel=32, kernel_size=3, padding=(1,1), maxpool=True)
        self.layer1 = conv_block(in_channel=32, out_channel=64, kernel_size=3, padding=(1,1), maxpool=True)
        self.layer2 = conv_block(in_channel=64, out_channel=128, kernel_size=3, padding=(1,1), maxpool=True)
        self.layer3 = conv_block(in_channel=128, out_channel=128, kernel_size=3, padding=(1,1), maxpool=True)
        self.layer4 = conv_block(in_channel=128, out_channel=256, kernel_size=3, padding=(1,1), maxpool=False)
        self.layer5 = conv_block(in_channel=256, out_channel=256, kernel_size=3, padding=(1,1), maxpool=False)
        self.layer6 = conv_block(in_channel=256, out_channel=16, kernel_size=3, padding=(1,1), maxpool=False)


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class LocalizingHead(nn.Module):
    def __init__(self):
        super(LocalizingHead, self).__init__()
        # input neurons: out_channel (num channels last conv_block) * downscaled resolution^2 (due to maxpooling)
        # hidden neurons: arbitrary (more -> more complex model)
        # output neurons: number of variables in output label
        self.head = nn.Sequential(nn.Linear(16 * (200 // (2**4))**2, 240), nn.ReLU(), nn.Linear(240, 6), nn.Sigmoid())

    def forward(self, x):
        batches, channels, height, width = x.shape
        x = x.contiguous()
        out = x.view(batches, channels * height * width)
        return self.head(out)

def conv_block(in_channel, out_channel, kernel_size, padding, maxpool=False, bias=True):
    if maxpool:
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, bias=bias, padding=padding),
                             nn.BatchNorm2d(out_channel),
                             nn.ReLU(),
                             nn.MaxPool2d(2, 2))
    else:
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, bias=bias, padding=padding),
                             nn.BatchNorm2d(out_channel),
                             nn.ReLU())
