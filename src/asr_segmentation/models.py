import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolution blocks with batch normalization, ReLU, and dropout."""

    def __init__(self, inputChannels, outputChannels, dropoutRate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropoutRate),
            nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropoutRate),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net for binary nuclei/cell segmentation."""

    def __init__(self, inputChannels=1, outputChannels=1, featureSizes=(32, 64, 128, 256), dropoutRate=0.3):
        super().__init__()

        self.downBlocks = nn.ModuleList()
        self.poolLayers = nn.ModuleList()

        currentChannels = inputChannels
        for featureCount in featureSizes:
            self.downBlocks.append(DoubleConv(currentChannels, featureCount, dropoutRate=dropoutRate))
            self.poolLayers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            currentChannels = featureCount

        self.bottleneck = DoubleConv(featureSizes[-1], featureSizes[-1] * 2, dropoutRate=dropoutRate)

        self.upSamples = nn.ModuleList()
        self.upBlocks = nn.ModuleList()

        currentChannels = featureSizes[-1] * 2
        for featureCount in featureSizes[::-1]:
            self.upSamples.append(nn.ConvTranspose2d(currentChannels, featureCount, kernel_size=2, stride=2))
            self.upBlocks.append(DoubleConv(featureCount * 2, featureCount, dropoutRate=dropoutRate))
            currentChannels = featureCount

        self.outputLayer = nn.Conv2d(featureSizes[0], outputChannels, kernel_size=1)

    def forward(self, x):
        skipConnections = []

        for downBlock, poolLayer in zip(self.downBlocks, self.poolLayers):
            x = downBlock(x)
            skipConnections.append(x)
            x = poolLayer(x)

        x = self.bottleneck(x)
        skipConnections = skipConnections[::-1]

        for blockIndex in range(len(self.upSamples)):
            x = self.upSamples[blockIndex](x)
            skipTensor = skipConnections[blockIndex]

            if x.shape[2:] != skipTensor.shape[2:]:
                x = F.interpolate(x, size=skipTensor.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([skipTensor, x], dim=1)
            x = self.upBlocks[blockIndex](x)

        return self.outputLayer(x)
