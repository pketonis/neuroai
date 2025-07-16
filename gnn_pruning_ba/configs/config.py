# configs/config.py

config = {
    # Graph parameters
    "n_clusters": 5,
    "cluster_size": 30,
    "m": 2,                   # BA preferential attachment parameter
    "p_noise": 0.33,          # Fraction of noisy (inter-community) edges

    # Node features
    "node_feat_dim": 16,

    # GCN architecture
    "hidden_dim": 32,

    # Training
    "epochs": 100,
    "learning_rate": 0.01,

    # Seed for reproducibility
    "random_seed": 42
}
