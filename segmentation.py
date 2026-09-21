"""
segmentation.py

Reusable segmentation core code for the ASR final project.

This file was made from the working segmentation_core_dev notebook.
It keeps the same student-friendly naming style used in the notebook:
    - pairFiles
    - pairB006
    - splitData
    - SegSet
    - makeLoader
    - UNet
    - SegLoss
    - trainNetwork
    - evaluateNetwork

Use this file in the active/passive comparison notebooks so the model,
preprocessing, loading, training, and evaluation code does not have to be
copied into every notebook.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tifffile as tiff
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


# ============================================================
# Reproducibility and device helpers
# ============================================================

def setSeed(seed=42):
    """
    Set random seeds so the same split/training run is easier to reproduce.

    Parameters
    ----------
    seed : int
        Random seed used for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def getDevice():
    """
    Return CUDA if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# File listing and image/mask pairing
# ============================================================

def fileStem(path):
    """
    Return filename without the extension.
    """
    return Path(path).stem


def listImages(imageDirectory):
    """
    List common image files inside a directory.

    This includes png, tif, tiff, jpg, and jpeg files.
    """
    imageDirectory = Path(imageDirectory)
    extensions = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"]

    imageFiles = []
    for extension in extensions:
        imageFiles.extend(imageDirectory.glob(extension))

    return sorted(imageFiles)


def cleanImageId(path):
    """
    Create a cleaned image ID for flexible pairing.

    This removes common words like mask/image/img from the filename stem.
    It is helpful when image and mask files are almost the same name but
    have small differences.
    """
    name = Path(path).stem.lower()

    removeWords = [
        "_mask", "-mask", "mask_", "mask-",
        "_image", "-image", "image_", "image-",
        "_img", "-img", "img_", "img-",
    ]

    for word in removeWords:
        name = name.replace(word, "")

    return name


def pairFiles(imageDirectory, maskDirectory, sourceName):
    """
    Pair image files with mask files using matching filename stems.

    This is the standard pairing function for datasets like BBBC039 and
    the combined dataset, where the image and mask filenames match.

    Parameters
    ----------
    imageDirectory : str or Path
        Folder containing input images.
    maskDirectory : str or Path
        Folder containing binary masks.
    sourceName : str
        Label for the dataset source, such as "BBBC039".

    Returns
    -------
    pairedTable : pandas.DataFrame
        Table with image_id, image_path, mask_path, and source columns.
    """
    imageFiles = listImages(Path(imageDirectory))
    maskFiles = listImages(Path(maskDirectory))

    imageMap = {fileStem(path): path for path in imageFiles}
    pairedRows = []

    for maskPath in maskFiles:
        imageId = fileStem(maskPath)

        if imageId in imageMap:
            pairedRows.append({
                "image_id": imageId,
                "image_path": str(imageMap[imageId]),
                "mask_path": str(maskPath),
                "source": sourceName,
            })

    # Flexible fallback if exact stems did not match.
    # This keeps the function useful when names differ by words like _mask.
    if len(pairedRows) == 0:
        imageMap = {cleanImageId(path): path for path in imageFiles}
        maskMap = {cleanImageId(path): path for path in maskFiles}
        commonIds = sorted(set(imageMap.keys()) & set(maskMap.keys()))

        for imageId in commonIds:
            pairedRows.append({
                "image_id": imageId,
                "image_path": str(imageMap[imageId]),
                "mask_path": str(maskMap[imageId]),
                "source": sourceName,
            })

    pairedTable = pd.DataFrame(pairedRows)

    if len(pairedTable) == 0:
        print("Found image files:", len(imageFiles))
        print("Found mask files:", len(maskFiles))
        print("Example image names:", [p.name for p in imageFiles[:5]])
        print("Example mask names:", [p.name for p in maskFiles[:5]])
        raise ValueError(f"No matched image-mask pairs found for source {sourceName}")

    pairedTable = pairedTable.sort_values(["source", "image_id"]).reset_index(drop=True)
    return pairedTable


def pairB006(imageDirectory, maskDirectory, sourceName="BBBC006", channelTag="_w1"):
    """
    Pair BBBC006 image files with masks.

    BBBC006 images can have channel information in the filename, such as _w1.
    The masks may use a shorter filename stem. This function looks for the
    image whose stem starts with the mask stem plus the selected channel tag.

    Parameters
    ----------
    imageDirectory : str or Path
        Folder containing BBBC006 images.
    maskDirectory : str or Path
        Folder containing BBBC006 masks.
    sourceName : str
        Source label, usually "BBBC006".
    channelTag : str
        Channel tag to pair with the mask, usually "_w1".

    Returns
    -------
    pairedTable : pandas.DataFrame
        Table with image_id, image_path, mask_path, and source columns.
    """
    imageFiles = listImages(Path(imageDirectory))
    maskFiles = listImages(Path(maskDirectory))

    pairedRows = []

    for maskPath in maskFiles:
        maskId = fileStem(maskPath)

        candidateImages = []
        for imagePath in imageFiles:
            imageId = fileStem(imagePath)
            if imageId.startswith(maskId + channelTag):
                candidateImages.append(imagePath)

        if len(candidateImages) == 1:
            pairedRows.append({
                "image_id": maskId,
                "image_path": str(candidateImages[0]),
                "mask_path": str(maskPath),
                "source": sourceName,
            })

    pairedTable = pd.DataFrame(pairedRows)

    if len(pairedTable) == 0:
        print("Found image files:", len(imageFiles))
        print("Found mask files:", len(maskFiles))
        print("Example image names:", [p.name for p in imageFiles[:5]])
        print("Example mask names:", [p.name for p in maskFiles[:5]])
        raise ValueError(f"No matched image-mask pairs found for source {sourceName}")

    pairedTable = pairedTable.sort_values(["source", "image_id"]).reset_index(drop=True)
    return pairedTable


# ============================================================
# Train/validation/test split
# ============================================================

def splitData(dataTable, seed=42, trainSize=0.70, validationSize=0.15, testSize=0.15):
    """
    Split a paired image/mask table into training, validation, and test tables.

    The output tables get a split column so each row remembers where it belongs.
    """
    totalSize = trainSize + validationSize + testSize

    if not np.isclose(totalSize, 1.0):
        raise ValueError("trainSize + validationSize + testSize must sum to 1.0")

    trainTable, tempTable = train_test_split(
        dataTable,
        test_size=(1.0 - trainSize),
        random_state=seed,
        shuffle=True,
    )

    relativeTestSize = testSize / (validationSize + testSize)

    validationTable, testTable = train_test_split(
        tempTable,
        test_size=relativeTestSize,
        random_state=seed,
        shuffle=True,
    )

    trainTable = trainTable.reset_index(drop=True).copy()
    validationTable = validationTable.reset_index(drop=True).copy()
    testTable = testTable.reset_index(drop=True).copy()

    trainTable["split"] = "train"
    validationTable["split"] = "validation"
    testTable["split"] = "test"

    return trainTable, validationTable, testTable


# Student-style alias from earlier notebook drafts.
def splitdata(df, seed=42, trainsize=0.70, validationsize=0.15, testsize=0.15):
    """
    Alias for splitData using earlier lowercase variable names.
    """
    return splitData(
        df,
        seed=seed,
        trainSize=trainsize,
        validationSize=validationsize,
        testSize=testsize,
    )


# ============================================================
# Image reading, normalization, and resizing
# ============================================================

def readImage(path):
    """
    Read a tif/tiff/png/jpg image into a NumPy array.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        imageArray = tiff.imread(str(path))
    else:
        imageArray = np.array(Image.open(path))

    return imageArray


def toTwoDimensional(imageArray):
    """
    Convert image array to 2D.

    If the image has a singleton dimension, it is squeezed.
    If the image has 3 dimensions, the first channel is used.
    """
    imageArray = np.squeeze(imageArray)

    if imageArray.ndim == 2:
        return imageArray

    if imageArray.ndim == 3:
        return imageArray[..., 0]

    raise ValueError(f"Unsupported image shape: {imageArray.shape}")


def normalizeGray(imageArray):
    """
    Normalize grayscale image values to the range 0 to 1 using min-max scaling.
    """
    imageArray = imageArray.astype(np.float32)

    lowestValue = imageArray.min()
    highestValue = imageArray.max()

    if highestValue > lowestValue:
        imageArray = (imageArray - lowestValue) / (highestValue - lowestValue)
    else:
        imageArray = np.zeros_like(imageArray, dtype=np.float32)

    return imageArray


def normalizeFluorescent(imageArray, lowPercentile=1.0, highPercentile=99.0):
    """
    Normalize fluorescent images using percentile clipping.

    This is useful when a few very bright pixels would otherwise dominate
    regular min-max scaling.
    """
    imageArray = imageArray.astype(np.float32)

    lowestValue = np.percentile(imageArray, lowPercentile)
    highestValue = np.percentile(imageArray, highPercentile)

    if highestValue > lowestValue:
        imageArray = (imageArray - lowestValue) / (highestValue - lowestValue)
        imageArray = np.clip(imageArray, 0, 1)
    else:
        imageArray = np.zeros_like(imageArray, dtype=np.float32)

    return imageArray


def resizeImage(imageArray, size=(256, 256)):
    """
    Resize a normalized image to the selected size.

    Bilinear interpolation is used because images have continuous intensity values.
    """
    pilImage = Image.fromarray((imageArray * 255).astype(np.uint8))
    resizedImage = pilImage.resize(size, Image.BILINEAR)
    resizedArray = np.array(resizedImage, dtype=np.float32) / 255.0

    return resizedArray


def resizeMask(maskArray, size=(256, 256)):
    """
    Resize a binary mask to the selected size.

    Nearest-neighbor interpolation is used so the mask stays binary.
    """
    pilMask = Image.fromarray((maskArray * 255).astype(np.uint8))
    resizedMask = pilMask.resize(size, Image.NEAREST)

    resizedArray = np.array(resizedMask, dtype=np.float32)
    resizedArray = (resizedArray > 0).astype(np.float32)

    return resizedArray


# Lowercase aliases from early drafts, in case a notebook still uses them.
readimg = readImage
make2d = toTwoDimensional
normalizationgray = normalizeGray
normalizationfluo = normalizeFluorescent
resizeimg = resizeImage
resizemask = resizeMask
listimgs = listImages


# ============================================================
# PyTorch datasets and dataloaders
# ============================================================

class SegSet(Dataset):
    """
    Segmentation dataset for grayscale or fluorescent image-mask pairs.

    Each item returned by this dataset is a dictionary with:
        image : torch tensor with shape [1, H, W]
        mask : torch tensor with shape [1, H, W]
        image_id : image identifier
        source : dataset source name
        split : train/validation/test label
        dataset_index : original dataset index
    """

    def __init__(self, dataTable, size=(256, 256), mode="auto"):
        self.dataTable = dataTable.reset_index(drop=True).copy()
        self.size = size
        self.mode = mode

    def __len__(self):
        return len(self.dataTable)

    def normalizeImage(self, imageArray, sourceName):
        """
        Choose gray or fluorescent normalization.
        """
        selectedMode = self.mode

        if selectedMode == "auto":
            if "039" in sourceName or "FLUO" in sourceName.upper():
                selectedMode = "fluo"
            else:
                selectedMode = "gray"

        if selectedMode == "gray":
            return normalizeGray(imageArray)

        if selectedMode == "fluo":
            return normalizeFluorescent(imageArray)

        raise ValueError(f"Unsupported mode: {self.mode}")

    def __getitem__(self, index):
        row = self.dataTable.iloc[index]

        imageArray = readImage(row["image_path"])
        maskArray = readImage(row["mask_path"])

        imageArray = toTwoDimensional(imageArray).astype(np.float32)
        maskArray = toTwoDimensional(maskArray).astype(np.float32)

        imageArray = self.normalizeImage(imageArray, row["source"])
        maskArray = (maskArray > 0).astype(np.float32)

        imageArray = resizeImage(imageArray, self.size)
        maskArray = resizeMask(maskArray, self.size)

        imageArray = np.expand_dims(imageArray, axis=0)
        maskArray = np.expand_dims(maskArray, axis=0)

        splitName = row["split"] if "split" in row.index else "na"

        return {
            "image": torch.tensor(imageArray, dtype=torch.float32),
            "mask": torch.tensor(maskArray, dtype=torch.float32),
            "image_id": row["image_id"],
            "source": row["source"],
            "split": splitName,
            "dataset_index": index,
        }


class IndexedSet(Dataset):
    """
    Thin wrapper that keeps only selected indices from a dataset.

    This is useful for active/passive learning notebooks where you train on
    a selected subset of images.
    """

    def __init__(self, dataset, selectedIndices):
        self.dataset = dataset
        self.selectedIndices = list(selectedIndices)

    def __len__(self):
        return len(self.selectedIndices)

    def __getitem__(self, index):
        datasetIndex = self.selectedIndices[index]
        item = self.dataset[datasetIndex]
        item["dataset_index"] = torch.tensor(datasetIndex, dtype=torch.long)
        return item


def makeLoader(dataset, batchSize=4, shuffle=False, numWorkers=0):
    """
    Create a PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
    )


# ============================================================
# U-Net model
# ============================================================

class DoubleConv(nn.Module):
    """
    Two convolution blocks with batch normalization, ReLU, and dropout.
    """

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
    """
    U-Net for binary nuclei/cell segmentation.

    The model takes a grayscale image tensor with shape [batch, 1, H, W]
    and returns mask logits with shape [batch, 1, H, W].
    """

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


# ============================================================
# Loss and metrics
# ============================================================

class SegLoss(nn.Module):
    """
    BCE-with-logits loss plus Dice loss for binary segmentation.
    """

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


def maskScores(predictionLogits, trueMasks, threshold=0.5, smooth=1e-6):
    """
    Compute batch Dice and IoU from logits and true masks.
    """
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


# ============================================================
# Evaluation and training
# ============================================================

def evaluateNetwork(network, dataLoader, lossFunction, device):
    """
    Evaluate a segmentation model on one dataloader.

    Returns a dictionary with average loss, Dice, and IoU.
    """
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
    """
    Train a segmentation model and track validation performance.

    Parameters
    ----------
    network : torch.nn.Module
        U-Net or another segmentation network.
    trainingLoader : DataLoader
        Training batches.
    validationLoader : DataLoader
        Validation batches.
    device : torch.device
        cuda or cpu.
    numEpochs : int
        Number of training epochs.
    learningRate : float
        Adam optimizer learning rate.
    savingPath : str or Path or None
        If given, the best model based on validation loss is saved here.
    printEvery : int or None
        Print batch progress every N batches. Use None to turn this off.

    Returns
    -------
    network : torch.nn.Module
        Trained network. If savingPath is provided, the best saved weights are reloaded.
    historyTable : pandas.DataFrame
        Epoch-level train loss, validation loss, Dice, and IoU.
    """
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


# ============================================================
# Plotting and result helpers
# ============================================================

def plotHistory(historyTable):
    """
    Plot training/validation loss and validation Dice/IoU.
    """
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
    """
    Show image, true mask, predicted mask, and prediction overlay.
    """
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
    """
    Save model weights, training history, and test results.
    """
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


# ============================================================
# MC dropout active learning helpers
# ============================================================

def enableDropoutForMCDropout(network):
    """
    Turn dropout layers on while keeping the rest of the network in eval mode.

    This is used for MC dropout. The idea is that the same image is passed
    through the model several times with dropout active. If the predictions
    vary a lot, the model is uncertain about that image.
    """
    network.eval()

    for module in network.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


def makeActiveSubsetMCDropout(dataset, trainedNetwork, device, warmupIndexes, numberOfSamples=120, batchSize=16, numMCSamples=5):
    """
    Create an active learning subset using MC dropout uncertainty.

    The final active subset includes:
        1. the warm-up indexes that were already labeled
        2. the most uncertain extra images selected by MC dropout

    The uncertainty score is the average pixel-wise variance across several
    stochastic predictions. Higher variance means the model is less certain.
    """
    trainedNetwork.eval()
    enableDropoutForMCDropout(trainedNetwork)

    loader = makeLoader(dataset, batchSize=batchSize, shuffle=False)

    uncertaintyRows = []
    currentIndex = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            mcPredictions = []

            for mcIndex in range(numMCSamples):
                predictionLogits = trainedNetwork(images)
                predictionProbabilities = torch.sigmoid(predictionLogits)
                mcPredictions.append(predictionProbabilities.cpu())

            mcPredictions = torch.stack(mcPredictions, dim=0)

            predictionVariance = torch.var(mcPredictions, dim=0)
            imageUncertainty = predictionVariance.mean(dim=(1, 2, 3)).numpy()

            for value in imageUncertainty:
                uncertaintyRows.append({
                    "index": currentIndex,
                    "uncertainty": float(value)
                })
                currentIndex += 1

    uncertaintyTable = pd.DataFrame(uncertaintyRows)

    uncertaintyTable = uncertaintyTable[
        ~uncertaintyTable["index"].isin(warmupIndexes)
    ]

    uncertaintyTable = uncertaintyTable.sort_values(
        "uncertainty",
        ascending=False
    ).reset_index(drop=True)

    numberToAdd = numberOfSamples - len(warmupIndexes)

    if numberToAdd < 0:
        raise ValueError("numberOfSamples must be greater than or equal to the number of warm-up indexes")

    newActiveIndexes = uncertaintyTable["index"].head(numberToAdd).tolist()
    selectedIndexes = list(warmupIndexes) + newActiveIndexes

    subset = torch.utils.data.Subset(dataset, selectedIndexes)

    return subset, selectedIndexes, uncertaintyTable
