# utils/idr_loss.py

def compute_idr_loss(activations, lambda_idr):
    loss = 0.0
    for act in activations:
        loss += act.abs().mean()
    return lambda_idr * loss
