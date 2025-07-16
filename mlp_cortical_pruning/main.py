# main.py

import torch
from configs.config import config
from models.sparse_mlp import SparseMLP
from data.mnist_loader import get_mnist_loaders
from train.trainer import train_with_pruning

def main():
    torch.manual_seed(config["random_seed"])

    train_loader, test_loader = get_mnist_loaders(
        batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"]
    )

    model = SparseMLP(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        num_classes=config["num_classes"]
    )

    metrics = train_with_pruning(model, config, train_loader, test_loader)

    print("Training complete.")
    print(f"Final test accuracy: {metrics['accuracy'][-1] * 100:.2f}%")

    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    from analysis.post_training_summary import plot_training_metrics
    plot_training_metrics(metrics)

if __name__ == "__main__":
    main()

