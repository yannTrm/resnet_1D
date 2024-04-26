# 1D ResNet for Sequence Data Classification

This repository contains an implementation of a 1D Residual Network (ResNet) designed for sequence data classification tasks. The architecture is based on the ResNet model and is specifically adapted for 1D sequence data.

## Overview

Residual Networks (ResNets) have revolutionized deep learning architectures by introducing skip connections, which help alleviate the vanishing gradient problem during training. This repository provides a specialized implementation of ResNet for processing 1D sequence data, such as time series or sequential sensor data.

### Key Components

- **Custom 1D ResNet architecture**: The architecture comprises 1D convolutional layers and residual blocks tailored for sequence data.
- **Residual blocks for efficient learning**: The inclusion of residual blocks allows for the training of deeper networks without suffering from degradation issues.
- **Configurations for ResNet models**: Different ResNet variants such as ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152 are provided to accommodate various complexities of sequence data classification tasks.

## Differences from 2D ResNet

In contrast to standard 2D ResNet architectures used for image classification, this 1D ResNet implementation is specifically designed for sequential data. The main differences include:

- **Convolutional Layers**: The ResNet for sequence data employs 1D convolutional layers instead of 2D convolutions used in image-based ResNets.
- **Input and Output Dimensions**: Input data for the 1D ResNet is typically represented as sequences of features over time, resulting in 1D input tensors and 1D output tensors, whereas image-based ResNets deal with 2D input tensors (images) and often produce 1D output tensors (class predictions).
- **Pooling Operations**: Pooling operations are typically performed along the temporal dimension in 1D ResNets, whereas 2D ResNets use spatial pooling operations.

## Advantages of ResNets for Sequence Data

ResNets offer several advantages for sequence data classification tasks:

- **Feature Reuse**: Skip connections enable the reuse of features from earlier layers in deeper layers, facilitating better feature representation.
- **Effective Training**: The presence of skip connections helps mitigate the vanishing gradient problem, making it easier to train very deep networks.
- **Hierarchical Feature Learning**: Residual blocks allow the network to learn hierarchical representations of input sequences, capturing both low-level and high-level features.

## Requirements

- Python 3.x
- PyTorch
- TorchVision

## Usage

To use the provided ResNet models for your sequence data classification tasks, follow these steps:

1. Import the necessary modules:

    ```python
    import torch
    import torch.nn as nn
    ```

2. Define your dataset and data loaders.

3. Choose the appropriate ResNet model (e.g., ResNet18_1D, ResNet34_1D) based on your requirements.

4. Instantiate the chosen model with appropriate input and output dimensions.

5. Train the model using your dataset.

6. Evaluate the trained model on test data.

## Examples

Example usage of the ResNet models can be found in the provided Jupyter notebooks or Python scripts.

## Contributing

Contributions to this repository are welcome! Feel free to open issues or pull requests with any improvements or bug fixes.


