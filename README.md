# neuroai

**Biologically Inspired Neural Networks with Pruning and Regularization**

This repository consolidates research and experiments on biologically inspired neural networks that implement *adaptive pruning*, *cortical structure*, and *information-based regularization*. The project explores how principles from neuroscience—such as synaptic pruning, cortical organization, and sparse activity—can be integrated into deep learning systems to enhance performance, efficiency, and interpretability.

---

## Project Overview

The project builds overparameterized models and progressively simplifies them during training using biologically motivated mechanisms:

- **Synaptic Pruning**: Iteratively removes unimportant connections based on dynamic weight magnitude or structured rules (e.g., community-based edge removal in GNNs).
- **Information Dimension Regularization (IDR)**: Encourages sparse internal codes by penalizing dense, high-dimensional neuron activations.
- **Lateral Inhibition**: Reduces redundancy by decorrelating activations within local neighborhoods.
- **Cortical Structure**:  
  - *Spatial cortical structure* for MLPs: neurons are organized in topographic maps where inhibition and pruning are spatially local.  
  - *Temporal cortical structure* for RNNs: hidden states and synapses evolve under structured dynamics over time, including memory traces and filtered activity.

Each architecture (MLP, RNN, GNN) is developed in its own subdirectory.

---

## Architectures

### `mlp/`
Fully connected networks for image classification (e.g., CIFAR-10).

- Overcomplete architecture pruned over time based on connection strength.
- Incorporates **spatial cortical topology**: neurons are arranged on a grid and inhibition is applied locally, simulating lateral inhibition in cortical sheets.
- Implements IDR to enforce low-dimensional internal activations.
- Evaluates accuracy and sparsity jointly.

### `rnn/`
Recurrent networks for temporal sequence modeling.

- Includes vanilla RNNs and GRUs.
- Uses **temporal cortical structure**: neurons are grouped together using cosine activation over a temporal signal
- IDR regularization applied to hidden states to compress dynamics.
- Connections are pruned based on their temporal activity relevance.

### `gnn/`
Graph neural networks for spatial-temporal and community-structured data.

- Edge dynamics governed by learnable Hebbian plasticity rules.
- Nodes maintain activity traces, and pruning targets edges between weakly interacting nodes.
- Tested on:
  - Synthetic graphs with known & unknown community structure.

---

## Goals and Motivation

This project models how neural circuits develop and specialize through experience:

- **Developmental Dynamics**: Start with dense, unspecialized networks that are shaped over time.
- **Energy Efficiency**: Use pruning and inhibition to reduce redundant computation.
- **Neuroscience-Inspired Computation**: Introduce biologically plausible mechanisms (e.g., topographic maps, Hebbian plasticity, trace memory).
- **Evaluation Metrics**: Combine traditional task performance (accuracy, loss) with biological interpretability (sparsity, dimensionality, structure retention).