# configs/config.py

config = {
    # General
    "random_seed": 42,
    "device": "cuda",  # Will fallback to CPU in main if unavailable

    # Data
    "batch_size": 64,
    "test_batch_size": 1000,

    # Training
    "initial_lr": 1e-5,
    "optimizer": "adam",
    "max_epochs_inner": 100,
    "convergence_patience": 3,
    "tolerance": 1e-3,
    "num_cycles": 3,
    "overall_patience": 3,

    # Loss
    "lambda_idr": 1e-2,  # IDR regularization strength

    # Pruning
    "similarity_threshold": 0.7,
    "top_k": 3,

    # Architecture
    "input_size": 784,
    "hidden_sizes": [256, 128],
    "num_classes": 10
}
