# utils/analysis.py

import numpy as np
import matplotlib.pyplot as plt

def compute_sparsity(layer):
    W = layer.weight.data
    total = W.numel()
    zeros = (W == 0).sum().item()
    return zeros / total

def get_weight_distribution(layer, threshold=1e-6):
    weights = layer.weight.data.cpu().numpy().flatten()
    nonzero_weights = weights[np.abs(weights) > threshold]
    return nonzero_weights

def neuron_death_count(layer):
    W = layer.weight.data
    dead = (W.abs().sum(dim=1) == 0).sum().item()
    total = W.size(0)
    return dead, total

def active_neuron_count(layer):
    W = layer.weight.data
    total = W.size(0)
    dead = (W.abs().sum(dim=1) == 0).sum().item()
    return total - dead, total

def plot_functional_purity(cos_sim, labels, title):
    within_sims = []
    between_sims = []

    if cos_sim.size > 0:
        n_neurons = cos_sim.shape[0]
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                if labels[i] == labels[j]:
                    within_sims.append(cos_sim[i, j])
                else:
                    between_sims.append(cos_sim[i, j])
        avg_within = np.mean(within_sims) if within_sims else 0
        avg_between = np.mean(between_sims) if between_sims else 0
    else:
        avg_within = avg_between = 0

    plt.figure()
    plt.bar(['Within Cluster', 'Between Cluster'], [avg_within, avg_between], color=['blue', 'orange'])
    plt.ylabel('Avg Cosine Similarity')
    plt.title(title)
    plt.show(block=False)
