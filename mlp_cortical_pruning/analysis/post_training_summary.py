# analysis/post_training_summary.py

import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(metrics):
    cycles = np.arange(1, len(metrics["train_loss"]) + 1)

    # --- Loss and Accuracy ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(cycles, metrics["train_loss"], label="Train Loss", marker='o')
    plt.plot(cycles, metrics["test_loss"], label="Test Loss", marker='o')
    plt.xlabel("Cycle")
    plt.ylabel("Loss")
    plt.title("Loss over Cycles")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cycles, [a * 100 for a in metrics["accuracy"]], label="Accuracy", color="green", marker='o')
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.tight_layout()
    plt.show()

    # --- Active Neurons ---
    plt.figure(figsize=(8, 4))
    plt.plot(cycles, metrics["active_neurons"]["fc1"], label="fc1", marker='o')
    plt.plot(cycles, metrics["active_neurons"]["fc2"], label="fc2", marker='o')
    plt.xlabel("Cycle")
    plt.ylabel("Active Neurons")
    plt.title("Active Neurons per Layer")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Sparsity ---
    plt.figure(figsize=(8, 4))
    plt.plot(cycles, [s * 100 for s in metrics["sparsity"]["fc1"]], label="fc1", marker='o')
    plt.plot(cycles, [s * 100 for s in metrics["sparsity"]["fc2"]], label="fc2", marker='o')
    plt.xlabel("Cycle")
    plt.ylabel("Sparsity (%)")
    plt.title("Layer Sparsity over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Pruned Neurons ---
    if "pruned_neurons" in metrics:
        plt.figure()
        plt.plot(cycles, metrics["pruned_neurons"], label="Pruned per Cycle", marker='o')
        plt.xlabel("Cycle")
        plt.ylabel("Neurons Pruned")
        plt.title("Pruning Events per Cycle")
        plt.tight_layout()
        plt.show()
