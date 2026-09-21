import numpy as np
import torch

from .loss import SegLoss, maskScores


def evaluateNetwork(network, dataLoader, lossFunction, device):
    """Evaluate a segmentation model on one dataloader."""
    network.eval()

    totalLoss = 0.0
    totalDice = 0.0
    totalIoU = 0.0
    totalBatches = 0

    with torch.no_grad():
        for batch in dataLoader:
            images = batch["image"].to(device)
            trueMasks = batch["mask"].to(device)

            predictionLogits = network(images)
            loss = lossFunction(predictionLogits, trueMasks)
            scoreTable = maskScores(predictionLogits, trueMasks)

            totalLoss += loss.item()
            totalDice += scoreTable["dice"]
            totalIoU += scoreTable["iou"]
            totalBatches += 1

    if totalBatches == 0:
        return {"loss": np.nan, "dice": np.nan, "iou": np.nan}

    return {
        "loss": totalLoss / totalBatches,
        "dice": totalDice / totalBatches,
        "iou": totalIoU / totalBatches,
    }
