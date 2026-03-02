import matplotlib.pyplot as plt

def plot_history(history, title="Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    
    # Plot Accuracy
    if "train_acc" in history:
        ax2.plot(history["train_acc"], label="Train Acc")
        ax2.plot(history["val_acc"], label="Val Acc")
        ax2.set_title(f"{title} - Accuracy")
        ax2.legend()
    
    plt.show()

def visualize_results(model, test_loader, device, n=3):
    model.eval()
    imgs, masks = next(iter(test_loader))
    imgs, masks = imgs[:n].to(device), masks[:n]
    
    with torch.no_grad():
        preds = torch.sigmoid(model(imgs)) # On applique sigmoid pour l'affichage
    
    plt.figure(figsize=(12, n*4))
    for i in range(n):
        plt.subplot(n, 3, i*3 + 1)
        plt.imshow(imgs[i].cpu().permute(1, 2, 0))
        plt.title("Image")
        plt.subplot(n, 3, i*3 + 2)
        plt.imshow(masks[i][0].cpu(), cmap='gray')
        plt.title("Masque Réel")
        plt.subplot(n, 3, i*3 + 3)
        plt.imshow(preds[i][0].cpu() > 0.5, cmap='gray')
        plt.title("Prédiction")
    plt.tight_layout()
    plt.show()