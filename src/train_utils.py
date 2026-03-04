import torch
from tqdm import tqdm
from src.metrics import pixel_accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    """
    Entraînement standard (Vanilla ou Phase 1) avec historique et tqdm.
    """
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    pbar = tqdm(range(epochs), desc="Training")
    
    for epoch in pbar:
        # --- PHASE TRAIN ---
        model.train()
        t_loss, t_acc = 0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            t_acc += pixel_accuracy(preds, masks)

        # --- PHASE VALIDATION ---
        model.eval()
        v_loss, v_acc = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds_v = model(imgs)
                v_loss += criterion(preds_v, masks).item()
                v_acc += pixel_accuracy(preds_v, masks)

        # Calcul des moyennes
        train_loss = t_loss / len(train_loader)
        val_loss = v_loss / len(val_loader)
        train_acc = t_acc / len(train_loader)
        val_acc = v_acc / len(val_loader)

        # Sauvegarde historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Mise à jour de la barre de progression
        pbar.set_postfix({"Loss": f"{train_loss:.4f}", "Val": f"{val_loss:.4f}"})
        
    return history

C'est une excellente idée. En uniformisant les deux fonctions, ton code sera beaucoup plus propre et tu pourras utiliser les mêmes outils de visualisation (plot_history) pour voir si ton modèle progresse bien dans sa compréhension des images de dents.

Voici la version "miroir" de train_model pour la reconstruction à mettre dans ton fichier src/train.py.

1. Modification de src/train.py
J'ai adapté la fonction pour qu'elle calcule la perte sur l'image elle-même (imgs vs preds) et j'ai retiré l'accuracy (qui n'a pas de sens pour des pixels continus) pour ne garder que la progression de la Loss (MSE).

Python
import torch
from tqdm import tqdm

def train_reconstruction(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    """
    Entraînement Phase 1 : Reconstruction. 
    La cible est l'image d'entrée (images == masks).
    """
    history = {"train_loss": [], "val_loss": []}
    
    pbar = tqdm(range(epochs), desc="Phase 1: Reconstruction")
    
    for epoch in pbar:
        # --- PHASE TRAIN ---
        model.train()
        t_loss = 0
        for imgs, _ in train_loader: # On ignore les vrais masques
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            # On compare l'image reconstruite à l'image d'origine
            loss = criterion(preds, imgs) 
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()

        # --- PHASE VALIDATION ---
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                preds_v = model(imgs)
                v_loss += criterion(preds_v, imgs).item()

        # Moyennes
        train_loss = t_loss / len(train_loader)
        val_loss = v_loss / len(val_loader)

        # Historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Mise à jour barre
        pbar.set_postfix({"Loss": f"{train_loss:.6f}", "Val": f"{val_loss:.6f}"})
        
    return history