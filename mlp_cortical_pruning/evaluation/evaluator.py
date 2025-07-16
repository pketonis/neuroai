# test/evaluator.py

import torch
import torch.nn.functional as F
from utils.idr_loss import compute_idr_loss

def evaluate_model(model, device, test_loader, lambda_idr):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, activations = model(data)
            ce_loss = F.cross_entropy(output, target, reduction='sum')
            idr_loss = compute_idr_loss(activations, lambda_idr)
            loss = ce_loss + idr_loss
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    total_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return total_loss, accuracy
