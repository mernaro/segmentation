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

def train_reconstruction(model, train_loader, val_loader, criterion, optimizer, device, epochs=30):
    """
    Spécifique à la Phase 0 : Reconstruction X -> X
    """
    history = {"train_loss": [], "val_loss": []}
    pbar = tqdm(range(epochs), desc="Reconstruction (Phase 0)")

    for epoch in pbar:
        model.train()
        t_loss = 0
        for imgs, _ in train_loader: # On ignore les masques ici
            imgs = imgs.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, imgs) # On compare l'image à elle-même
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                v_loss += criterion(model(imgs), imgs).item()

        history["train_loss"].append(t_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))
        pbar.set_postfix({"MSE": f"{t_loss/len(train_loader):.4f}"})

    return history