# MLP with Cortical Pruning

This module implements a biologically inspired overparameterized multilayer perceptron (MLP) trained on MNIST with **similarity-based pruning** and **information dimension regularization (IDR)**.

## Key Features

- **Spatial Cortical Structure**: Neurons are arranged in topographic clusters. Cosine similarity is used to identify and prune redundant neurons within each cluster.
- **Information Dimension Regularization (IDR)**: An L1 penalty on activations encourages sparse, low-dimensional representations.
- **Pruning**: After each training cycle, redundant neurons are identified and removed by zeroing their weights and biases.
- **Train-Until-Convergence Cycles**: The network is trained until convergence in each pruning cycle before applying pruning and continuing.