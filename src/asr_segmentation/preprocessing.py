from pathlib import Path

import numpy as np
from PIL import Image
import tifffile as tiff


def readImage(path):
    """Read a tif/tiff/png/jpg image into a NumPy array."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        imageArray = tiff.imread(str(path))
    else:
        imageArray = np.array(Image.open(path))

    return imageArray


def toTwoDimensional(imageArray):
    """Convert image array to 2D."""
    imageArray = np.squeeze(imageArray)

    if imageArray.ndim == 2:
        return imageArray

    if imageArray.ndim == 3:
        return imageArray[..., 0]

    raise ValueError(f"Unsupported image shape: {imageArray.shape}")


def normalizeGray(imageArray):
    """Normalize grayscale image values to the range 0 to 1 using min-max scaling."""
    imageArray = imageArray.astype(np.float32)

    lowestValue = imageArray.min()
    highestValue = imageArray.max()

    if highestValue > lowestValue:
        imageArray = (imageArray - lowestValue) / (highestValue - lowestValue)
    else:
        imageArray = np.zeros_like(imageArray, dtype=np.float32)

    return imageArray


def normalizeFluorescent(imageArray, lowPercentile=1.0, highPercentile=99.0):
    """Normalize fluorescent images using percentile clipping."""
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
    """Resize a normalized image to the selected size."""
    pilImage = Image.fromarray((imageArray * 255).astype(np.uint8))
    resizedImage = pilImage.resize(size, Image.BILINEAR)
    resizedArray = np.array(resizedImage, dtype=np.float32) / 255.0

    return resizedArray


def resizeMask(maskArray, size=(256, 256)):
    """Resize a binary mask to the selected size."""
    pilMask = Image.fromarray((maskArray * 255).astype(np.uint8))
    resizedMask = pilMask.resize(size, Image.NEAREST)

    resizedArray = np.array(resizedMask, dtype=np.float32)
    resizedArray = (resizedArray > 0).astype(np.float32)

    return resizedArray


readimg = readImage
make2d = toTwoDimensional
normalizationgray = normalizeGray
normalizationfluo = normalizeFluorescent
resizeimg = resizeImage
resizemask = resizeMask


def listimgs(directory):
    """Return sorted image files with common microscopy extensions."""
    directory = Path(directory)
    extensions = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    )
