from pathlib import Path
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def setSeed(seed=42):
    """Set random seeds so the same split/training run is easier to reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def getDevice():
    """Return CUDA if available, otherwise CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fileStem(path):
    """Return filename without the extension."""
    return Path(path).stem


def listImages(imageDirectory):
    """List common image files inside a directory."""
    imageDirectory = Path(imageDirectory)
    extensions = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"]

    imageFiles = []
    for extension in extensions:
        imageFiles.extend(imageDirectory.glob(extension))

    return sorted(imageFiles)


def cleanImageId(path):
    """Create a cleaned image ID for flexible pairing."""
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
    """Pair image files with mask files using matching filename stems."""
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
    """Pair BBBC006 image files with masks."""
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


def splitData(dataTable, seed=42, trainSize=0.70, validationSize=0.15, testSize=0.15):
    """Split a paired image/mask table into training, validation, and test tables."""
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


def splitdata(df, seed=42, trainsize=0.70, validationsize=0.15, testsize=0.15):
    """Alias for splitData using earlier lowercase variable names."""
    return splitData(
        df,
        seed=seed,
        trainSize=trainsize,
        validationSize=validationsize,
        testSize=testsize,
    )
