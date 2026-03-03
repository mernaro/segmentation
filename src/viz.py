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

def visualize_results(model, test_loader, device, n_images=5):
    model.eval()
    images, masks = next(iter(test_loader)) # On prend un batch du test_loader
    images = images[:n_images].to(device)
    masks = masks[:n_images].to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = (outputs > 0.5).float() # Seuil de binarisation

    plt.figure(figsize=(15, n_images * 3))
    
    for i in range(n_images):
        # Image originale
        plt.subplot(n_images, 3, i * 3 + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title("Radiographie (Input)")
        plt.axis('off')

        # Prédiction
        plt.subplot(n_images, 3, i * 3 + 2)
        plt.imshow(preds[i].cpu().squeeze(), cmap='gray')
        plt.title("Segmentation (Modèle)")
        plt.axis('off')

        # Vérité terrain
        plt.subplot(n_images, 3, i * 3 + 3)
        plt.imshow(masks[i].cpu().squeeze(), cmap='gray')
        plt.title("Masque Réel (Ground Truth)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()