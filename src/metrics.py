import torch
import numpy as np

def pixel_accuracy(output, mask):
    with torch.no_grad():
        # On passe en binaire (0 ou 1)
        output = (output > 0.5).float()
        correct = (output == mask).float()
        accuracy = correct.sum() / correct.numel()
    return accuracy.item()

def calculate_metrics(model, x_test, y_test, device="cuda"):
    model.eval()
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for i in range(len(x_test)):
            img = torch.from_numpy(x_test[i:i+1]).float().to(device)
            mask = torch.from_numpy(y_test[i:i+1]).float().to(device)
            
            pred = model(img)
            pred = (pred > 0.5).float()
            
            # Calcul Dice
            intersection = (pred * mask).sum()
            dice = (2. * intersection) / (pred.sum() + mask.sum() + 1e-8)
            
            # Calcul IoU
            union = (pred + mask).clamp(0, 1).sum()
            iou = intersection / (union + 1e-8)
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            
    return np.mean(dice_scores), np.mean(iou_scores)