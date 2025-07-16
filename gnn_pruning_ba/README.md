# GNN Pruning on BA-Community Graphs

This module implements graph pruning based on inter-community edge removal using synthetic **Barabási-Albert (BA) community graphs**. The pruning is evaluated using a **2-layer GCN** and compared to a non-pruned baseline.

## Features

- BA graph generation with controllable cluster size and noise edges
- Intra- vs inter-community pruning mask based on ground-truth labels
- Custom `GCNMasked` model with masked message passing
- Evaluation of pruning effect on classification accuracy
- Functional metrics: FP/FN and visual pruning inspection