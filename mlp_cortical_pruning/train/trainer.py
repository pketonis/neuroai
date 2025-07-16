# train/trainer.py

import copy
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.idr_loss import compute_idr_loss
from evaluation.evaluator import evaluate_model
from utils.pruning import (
    prune_model,
    compute_layer_cosine_similarity,
    cluster_neurons,
)
from utils.analysis import (
    compute_sparsity,
    get_weight_distribution,
    neuron_death_count,
    active_neuron_count,
    plot_functional_purity,
)


def train_model(model, device, train_loader, optimizer, epoch, lambda_idr):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, activations = model(data)
        ce_loss = torch.nn.functional.cross_entropy(output, target)
        idr_loss = compute_idr_loss(activations, lambda_idr)
        loss = ce_loss + idr_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def train_with_pruning(model, config, train_loader, test_loader):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["initial_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    best_loss = float('inf')
    global_epoch_counter = 0
    overall_patience_counter = 0
    best_model_state = None

    metrics = {
        "train_loss": [],
        "test_loss": [],
        "accuracy": [],
        "sparsity": {"fc1": [], "fc2": []},
        "active_neurons": {"fc1": [], "fc2": []},
        "clusters": {"fc1": [], "fc2": []},
        "cluster_sizes": {"fc1": [], "fc2": []},
        "weights": {"fc1": [], "fc2": []},
        "pruned_neurons": []
    }

    for cycle in range(1, config["num_cycles"] + 1):
        print(f"\n=== Cycle {cycle} ===")
        inner_patience = 0
        best_loss_cycle = float('inf')
        epoch_in_cycle = 0

        while inner_patience < config["convergence_patience"] and epoch_in_cycle < config["max_epochs_inner"]:
            epoch_in_cycle += 1
            global_epoch_counter += 1
            train_loss = train_model(
                model, device, train_loader, optimizer, global_epoch_counter, config["lambda_idr"]
            )
            if train_loss < best_loss_cycle - config["tolerance"]:
                best_loss_cycle = train_loss
                inner_patience = 0
            else:
                inner_patience += 1
            print(f"Cycle {cycle}, Epoch {epoch_in_cycle}, Train Loss: {train_loss:.4f}")

        metrics["train_loss"].append(best_loss_cycle)

        test_loss, acc = evaluate_model(model, device, test_loader, config["lambda_idr"])
        metrics["test_loss"].append(test_loss)
        metrics["accuracy"].append(acc)
        scheduler.step(test_loss)

        for layer_name in ["fc1", "fc2"]:
            layer = getattr(model, layer_name)
            sparsity = compute_sparsity(layer)
            active, _ = active_neuron_count(layer)
            weights = get_weight_distribution(layer)

            metrics["sparsity"][layer_name].append(sparsity)
            metrics["active_neurons"][layer_name].append(active)
            metrics["weights"][layer_name].append(weights)

        with torch.no_grad():
            data, _ = next(iter(test_loader))
            data = data.to(device)
            _, activations = model(data)
            cos_sim = compute_layer_cosine_similarity(activations[0])
            labels = cluster_neurons(cos_sim, config["similarity_threshold"])
            plot_functional_purity(cos_sim, labels, f"Functional Purity BEFORE Pruning (Cycle {cycle})")

        pruned_summary, clustering_info = prune_model(
            model, device, test_loader,
            similarity_threshold=config["similarity_threshold"],
            top_k=config["top_k"]
        )
        metrics["pruned_neurons"].append(sum(pruned_summary.values()))

        for layer_key in ["fc1", "fc2"]:
            labels = clustering_info[layer_key]["labels"]
            valid_labels = labels[labels != -1]
            if valid_labels.size > 0:
                unique_labels, counts = np.unique(valid_labels, return_counts=True)
                num_clusters = len(unique_labels)
            else:
                num_clusters = 0
                counts = np.array([])
            metrics["clusters"][layer_key].append(num_clusters)
            metrics["cluster_sizes"][layer_key].append(counts)

        optimizer = optim.Adam(model.parameters(), lr=config["initial_lr"])

        if clustering_info["fc1"]["cos_sim"].size:
            plot_functional_purity(
                clustering_info["fc1"]["cos_sim"],
                clustering_info["fc1"]["labels"],
                title=f"Functional Purity AFTER Pruning (Cycle {cycle})"
            )

        if test_loss < best_loss:
            best_loss = test_loss
            overall_patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Validation loss improved to {best_loss:.4f}")
        else:
            overall_patience_counter += 1
            print(f"No improvement in validation loss for {overall_patience_counter} cycle(s).")
            if overall_patience_counter >= config["overall_patience"]:
                print(f"Early stopping triggered.")
                break

        orig_fc1, orig_fc2 = config["hidden_sizes"]
        if (metrics["active_neurons"]["fc1"][-1] <= 0.1 * orig_fc1 and
            metrics["active_neurons"]["fc2"][-1] <= 0.1 * orig_fc2):
            print("Stop: Active neurons reached 10% of original count.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model restored.")

    # --- Final reporting ---
    for layer_name in ["fc1", "fc2"]:
        layer = getattr(model, layer_name)
        active, total = active_neuron_count(layer)
        sparsity = compute_sparsity(layer)
        print(f"[FINAL] {layer_name}: {active}/{total} active neurons | {sparsity * 100:.2f}% sparsity")

    # --- Final functional purity ---
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        _, activations = model(data)
        cos_sim_final = compute_layer_cosine_similarity(activations[0])
        labels_final = cluster_neurons(cos_sim_final, config["similarity_threshold"])
        plot_functional_purity(cos_sim_final, labels_final, "Functional Purity at Final Model State")

    # --- Final weight histograms ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(get_weight_distribution(model.fc1), bins=50, color='blue', alpha=0.7)
    plt.title("Final Weight Distribution - fc1 (nonzero)")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(get_weight_distribution(model.fc2), bins=50, color='orange', alpha=0.7)
    plt.title("Final Weight Distribution - fc2 (nonzero)")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    return metrics
