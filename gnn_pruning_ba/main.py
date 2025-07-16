# main.py

from configs.config import config
from data.ba_generator import generate_ba_community
from train.trainer import train_and_evaluate
from utils.pruning import create_edge_mask, evaluate_pruning
from utils.visualization import plot_pruned_graph

def main():
    print("Generating BA-community graph...")
    data = generate_ba_community(
        n_clusters=config["n_clusters"],
        cluster_size=config["cluster_size"],
        m=config["m"],
        p_noise=config["p_noise"]
    )

    print("\nTraining baseline GCN (no pruning)...")
    acc_baseline = train_and_evaluate(data, edge_mask=None, config=config)

    print("\nCreating inter-community pruning mask...")
    edge_mask = create_edge_mask(data)

    print("\nTraining pruned GCN...")
    acc_pruned = train_and_evaluate(data, edge_mask=edge_mask, config=config)

    print(f"\nBaseline Accuracy: {acc_baseline:.4f}")
    print(f"Pruned Accuracy:   {acc_pruned:.4f}")

    print("\nEvaluating pruning correctness...")
    evaluate_pruning(data, edge_mask)

    print("\nVisualizing pruned graph...")
    plot_pruned_graph(data, edge_mask)

if __name__ == "__main__":
    main()
