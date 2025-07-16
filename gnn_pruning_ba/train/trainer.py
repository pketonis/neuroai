# train/trainer.py

import torch
import torch.nn as nn
from models.gcn_masked import GCNMasked

def train_and_evaluate(data, edge_mask, config):
    torch.manual_seed(config["random_seed"])

    # Random initial node features
    x = torch.rand(data.num_nodes, config["node_feat_dim"])

    model = GCNMasked(
        in_dim=config["node_feat_dim"],
        hidden_dim=config["hidden_dim"],
        out_dim=data.y.max().item() + 1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        model.train()
        out = model(x, data.edge_index, edge_mask=edge_mask)
        loss = loss_fn(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x, data.edge_index, edge_mask=edge_mask)
        pred = logits.argmax(dim=1)
        acc = (pred == data.y).float().mean().item()

    return acc
