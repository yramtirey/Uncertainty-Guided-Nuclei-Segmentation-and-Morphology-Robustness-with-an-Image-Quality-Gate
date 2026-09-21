from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .evaluation import evaluateNetwork
from .loss import SegLoss


def trainNetwork(
    network,
    trainingLoader,
    validationLoader,
    device,
    numEpochs=5,
    learningRate=1e-3,
    savingPath=None,
    printEvery=10,
):
    """Train a segmentation model and track validation performance."""
    optimizer = torch.optim.Adam(network.parameters(), lr=learningRate)
    lossFunction = SegLoss()

    bestValidationLoss = float("inf")
    historyRows = []

    for epochIndex in range(numEpochs):
        network.train()

        totalTrainingLoss = 0.0
        totalTrainingBatches = 0

        for batchIndex, batch in enumerate(trainingLoader):
            images = batch["image"].to(device)
            trueMasks = batch["mask"].to(device)

            optimizer.zero_grad()
            predictionLogits = network(images)
            loss = lossFunction(predictionLogits, trueMasks)
            loss.backward()
            optimizer.step()

            if printEvery is not None and batchIndex % printEvery == 0:
                print(
                    f"Epoch {epochIndex + 1}/{numEpochs} | "
                    f"Batch {batchIndex + 1}/{len(trainingLoader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

            totalTrainingLoss += loss.item()
            totalTrainingBatches += 1

        averageTrainingLoss = totalTrainingLoss / max(totalTrainingBatches, 1)

        validationScores = evaluateNetwork(
            network=network,
            dataLoader=validationLoader,
            lossFunction=lossFunction,
            device=device,
        )

        historyRows.append({
            "epoch": epochIndex + 1,
            "trainLoss": averageTrainingLoss,
            "validationLoss": validationScores["loss"],
            "validationDice": validationScores["dice"],
            "validationIoU": validationScores["iou"],
        })

        print(
            f"Epoch {epochIndex + 1}/{numEpochs} | "
            f"Train Loss: {averageTrainingLoss:.4f} | "
            f"Validation Loss: {validationScores['loss']:.4f} | "
            f"Validation Dice: {validationScores['dice']:.4f} | "
            f"Validation IoU: {validationScores['iou']:.4f}",
            flush=True,
        )

        if validationScores["loss"] < bestValidationLoss:
            bestValidationLoss = validationScores["loss"]

            if savingPath is not None:
                savingPath = Path(savingPath)
                savingPath.parent.mkdir(parents=True, exist_ok=True)
                torch.save(network.state_dict(), savingPath)

    if savingPath is not None and Path(savingPath).exists():
        network.load_state_dict(torch.load(savingPath, map_location=device))

    historyTable = pd.DataFrame(historyRows)
    return network, historyTable


def plotHistory(historyTable):
    """Plot training/validation loss and validation Dice/IoU."""
    plt.figure(figsize=(8, 4))
    plt.plot(historyTable["epoch"], historyTable["trainLoss"], label="Train Loss")
    plt.plot(historyTable["epoch"], historyTable["validationLoss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(historyTable["epoch"], historyTable["validationDice"], label="Validation Dice")
    plt.plot(historyTable["epoch"], historyTable["validationIoU"], label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Dice and IoU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def showPredictions(network, dataset, device, numberOfImages=3, threshold=0.5):
    """Show image, true mask, predicted mask, and prediction overlay."""
    network.eval()

    numberOfImages = min(numberOfImages, len(dataset))
    plt.figure(figsize=(12, 4 * numberOfImages))

    for i in range(numberOfImages):
        sample = dataset[i]

        image = sample["image"].unsqueeze(0).to(device)
        trueMask = sample["mask"].squeeze().cpu().numpy()

        with torch.no_grad():
            predictionLogits = network(image)
            predictionProbabilities = torch.sigmoid(predictionLogits)
            predictedMask = predictionProbabilities.squeeze().cpu().numpy()

        predictedBinaryMask = (predictedMask > threshold).astype(np.float32)
        imageToShow = sample["image"].squeeze().cpu().numpy()

        plt.subplot(numberOfImages, 4, i * 4 + 1)
        plt.imshow(imageToShow, cmap="gray")
        plt.title("Image")
        plt.axis("off")

        plt.subplot(numberOfImages, 4, i * 4 + 2)
        plt.imshow(trueMask, cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

        plt.subplot(numberOfImages, 4, i * 4 + 3)
        plt.imshow(predictedBinaryMask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(numberOfImages, 4, i * 4 + 4)
        plt.imshow(imageToShow, cmap="gray")
        plt.imshow(predictedBinaryMask, alpha=0.4)
        plt.title("Prediction Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def saveResults(trainedNetwork, historyTable, testResults, outputDirectory, modelName="segmentation_unet_final_model.pth"):
    """Save model weights, training history, and test results."""
    outputDirectory = Path(outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)

    modelPath = outputDirectory / modelName
    historyPath = outputDirectory / "segmentation_training_history.csv"
    testResultsPath = outputDirectory / "segmentation_test_results.csv"

    torch.save(trainedNetwork.state_dict(), modelPath)
    historyTable.to_csv(historyPath, index=False)
    pd.DataFrame([testResults]).to_csv(testResultsPath, index=False)

    print("Saved model to:", modelPath)
    print("Saved history to:", historyPath)
    print("Saved test results to:", testResultsPath)

    return modelPath, historyPath, testResultsPath
