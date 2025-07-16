# utils/pruning.py

import torch

def create_edge_mask(data):
    src = data.edge_index[0]
    tgt = data.edge_index[1]
    same_community = data.y[src] == data.y[tgt]
    num_pruned = (~same_community).sum().item()
    num_total = same_community.size(0)
    print(f"Pruned {num_pruned}/{num_total} inter-community edges ({100 * num_pruned / num_total:.2f}%)")
    return same_community  # Boolean mask: True = keep edge

def evaluate_pruning(data, edge_mask):
    src = data.edge_index[0]
    tgt = data.edge_index[1]
    y = data.y
    same = y[src] == y[tgt]

    TP = ((same == True) & (edge_mask == True)).sum().item()
    TN = ((same == False) & (edge_mask == False)).sum().item()
    FP = ((same == False) & (edge_mask == True)).sum().item()
    FN = ((same == True) & (edge_mask == False)).sum().item()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("Pruning Evaluation:")
    print(f"  FP (inter kept):   {FP}")
    print(f"  FN (intra pruned): {FN}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
