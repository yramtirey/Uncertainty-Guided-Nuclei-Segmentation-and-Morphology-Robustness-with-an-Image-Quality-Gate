import pandas as pd
import torch
from torch.utils.data import Subset

from .preprocessing import listimgs


def enableDropoutForMCDropout(network):
    """Turn dropout layers on while keeping the rest of the network in eval mode."""
    network.eval()

    for module in network.modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()


def makeActiveSubsetMCDropout(dataset, trainedNetwork, device, warmupIndexes, numberOfSamples=120, batchSize=16, numMCSamples=5):
    """Create an active learning subset using MC dropout uncertainty."""
    trainedNetwork.eval()
    enableDropoutForMCDropout(trainedNetwork)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)

    uncertaintyRows = []
    currentIndex = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            mcPredictions = []

            for _ in range(numMCSamples):
                predictionLogits = trainedNetwork(images)
                predictionProbabilities = torch.sigmoid(predictionLogits)
                mcPredictions.append(predictionProbabilities.cpu())

            mcPredictions = torch.stack(mcPredictions, dim=0)

            predictionVariance = torch.var(mcPredictions, dim=0)
            imageUncertainty = predictionVariance.mean(dim=(1, 2, 3)).numpy()

            for value in imageUncertainty:
                uncertaintyRows.append({
                    "index": currentIndex,
                    "uncertainty": float(value),
                })
                currentIndex += 1

    uncertaintyTable = pd.DataFrame(uncertaintyRows)

    uncertaintyTable = uncertaintyTable[
        ~uncertaintyTable["index"].isin(warmupIndexes)
    ]

    uncertaintyTable = uncertaintyTable.sort_values(
        "uncertainty",
        ascending=False,
    ).reset_index(drop=True)

    numberToAdd = numberOfSamples - len(warmupIndexes)

    if numberToAdd < 0:
        raise ValueError("numberOfSamples must be greater than or equal to the number of warm-up indexes")

    newActiveIndexes = uncertaintyTable["index"].head(numberToAdd).tolist()
    selectedIndexes = list(warmupIndexes) + newActiveIndexes

    subset = Subset(dataset, selectedIndexes)

    return subset, selectedIndexes, uncertaintyTable
