# utils/pruning.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import torch

def compute_layer_cosine_similarity(activations):
    act_np = activations.detach().cpu().numpy()
    neuron_acts = act_np.T
    return cosine_similarity(neuron_acts)

def cluster_neurons(cosine_sim, similarity_threshold=0.7):
    clustering = AgglomerativeClustering(metric='precomputed',
                                         linkage='average',
                                         distance_threshold=1 - similarity_threshold,
                                         n_clusters=None)
    labels = clustering.fit_predict(1 - cosine_sim)
    return labels

def prune_layer(layer, activations, similarity_threshold=0.8, top_k=3, eps=1e-8):
    act_np = activations.detach().cpu().numpy()
    neuron_norms = np.linalg.norm(act_np, axis=0)
    alive_indices = np.where(neuron_norms > eps)[0]
    n_total = act_np.shape[1]

    labels_full = -1 * np.ones(n_total, dtype=int)
    pruned_neurons = []
    cos_sim_alive = np.array([])

    if len(alive_indices) > 0:
        alive_activations = activations[:, alive_indices]
        cos_sim = compute_layer_cosine_similarity(alive_activations)
        labels = cluster_neurons(cos_sim, similarity_threshold)
        cos_sim_alive = cos_sim.copy()
        for idx, orig_idx in enumerate(alive_indices):
            labels_full[orig_idx] = labels[idx]
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            rel_indices = np.where(labels == lbl)[0]
            orig_indices = alive_indices[rel_indices]
            if len(orig_indices) > top_k:
                W = layer.weight.data
                norms = [(i, W[i].abs().sum().item()) for i in orig_indices]
                sorted_norms = sorted(norms, key=lambda x: x[1], reverse=True)
                for tup in sorted_norms[top_k:]:
                    i = tup[0]
                    layer.weight.data[i].zero_()
                    if layer.bias is not None:
                        layer.bias.data[i].zero_()
                    pruned_neurons.append(i)
    return pruned_neurons, labels_full, cos_sim_alive

def prune_model(model, device, val_loader, similarity_threshold=0.8, top_k=3):
    model.eval()
    pruned_summary = {}
    clustering_info = {}

    with torch.no_grad():
        data, _ = next(iter(val_loader))
        data = data.to(device)
        _, activations = model(data)
        
        pruned_fc1, labels_fc1, cos_sim_fc1 = prune_layer(model.fc1, activations[0],
                                                           similarity_threshold, top_k)
        pruned_summary['fc1'] = len(pruned_fc1)
        clustering_info['fc1'] = {'labels': labels_fc1, 'cos_sim': cos_sim_fc1}

        pruned_fc2, labels_fc2, cos_sim_fc2 = prune_layer(model.fc2, activations[1],
                                                           similarity_threshold, top_k)
        pruned_summary['fc2'] = len(pruned_fc2)
        clustering_info['fc2'] = {'labels': labels_fc2, 'cos_sim': cos_sim_fc2}

    return pruned_summary, clustering_info
