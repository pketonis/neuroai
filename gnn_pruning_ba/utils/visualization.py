# utils/visualization.py

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def plot_pruned_graph(data, edge_mask):
    G = to_networkx(data, to_undirected=True)
    communities = data.y.cpu().numpy().tolist()

    kept_edges = []
    pruned_edges = []
    edge_index = data.edge_index.cpu().numpy()

    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        edge = (u, v) if u < v else (v, u)
        if edge_mask[i]:
            kept_edges.append(edge)
        else:
            pruned_edges.append(edge)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=communities, cmap='Set2', node_size=50)
    nx.draw_networkx_edges(G, pos, edgelist=kept_edges, edge_color='gray', alpha=0.4)
    nx.draw_networkx_edges(G, pos, edgelist=pruned_edges, edge_color='red', alpha=0.7, width=1)
    plt.title("BA-Community Graph\nGray = Kept Edges, Red = Pruned Inter-Community Edges")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
