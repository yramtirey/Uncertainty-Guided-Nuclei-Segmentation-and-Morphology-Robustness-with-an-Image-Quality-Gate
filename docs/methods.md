# Methods

This project uses a U-Net architecture for binary microscopy segmentation with a binary cross-entropy plus Dice loss objective.

## Data preparation

The notebooks pair image and mask files by dataset, then split each dataset into train, validation, and test partitions using a fixed seed. Images are normalized to a consistent grayscale or fluorescent intensity scale, reduced to two dimensions when needed, and resized to a fixed spatial resolution before training.

## Model

The model follows a standard U-Net design built from repeated convolution blocks, pooling, skip connections, and transposed-convolution upsampling. It produces a single-channel logit map for the binary segmentation mask.

## Loss and evaluation

The loss is BCEWithLogitsLoss plus a Dice term. Performance is reported using Dice score and IoU on the validation and test sets.

## Active learning setup

The active-learning variant keeps a warm-up subset and then selects additional images using MC dropout uncertainty. Each selected image is passed through the model multiple times with dropout enabled; the variance across predictions defines uncertainty. The model then trains on the warm-up plus the highest-uncertainty unlabeled examples under a matched labeling budget.

## Saved results

The canonical results are preserved in the outputs directory. The summary values recorded in the project outputs are the basis for the final project writeup.
