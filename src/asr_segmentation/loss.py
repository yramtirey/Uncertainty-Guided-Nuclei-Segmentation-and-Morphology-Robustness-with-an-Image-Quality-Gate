import torch
import torch.nn as nn


def maskScores(predictionLogits, trueMasks, threshold=0.5, smooth=1e-6):
    """Compute batch Dice and IoU from logits and true masks."""
    predictionProbabilities = torch.sigmoid(predictionLogits)
    predictedMasks = (predictionProbabilities > threshold).float()

    predictedMasks = predictedMasks.view(predictedMasks.size(0), -1)
    trueMasks = trueMasks.view(trueMasks.size(0), -1)

    intersection = (predictedMasks * trueMasks).sum(dim=1)
    union = predictedMasks.sum(dim=1) + trueMasks.sum(dim=1) - intersection

    diceScore = (
        (2.0 * intersection + smooth) /
        (predictedMasks.sum(dim=1) + trueMasks.sum(dim=1) + smooth)
    ).mean().item()

    iouScore = (
        (intersection + smooth) /
        (union + smooth)
    ).mean().item()

    return {
        "dice": diceScore,
        "iou": iouScore,
    }


class SegLoss(nn.Module):
    """BCE-with-logits loss plus Dice loss for binary segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.binaryCrossEntropy = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, predictionLogits, trueMasks):
        binaryLoss = self.binaryCrossEntropy(predictionLogits, trueMasks)

        predictionProbabilities = torch.sigmoid(predictionLogits)
        predictionProbabilities = predictionProbabilities.view(predictionProbabilities.size(0), -1)
        trueMasks = trueMasks.view(trueMasks.size(0), -1)

        intersection = (predictionProbabilities * trueMasks).sum(dim=1)
        diceScore = (2.0 * intersection + self.smooth) / (
            predictionProbabilities.sum(dim=1) + trueMasks.sum(dim=1) + self.smooth
        )
        diceLoss = 1.0 - diceScore.mean()

        return binaryLoss + diceLoss
