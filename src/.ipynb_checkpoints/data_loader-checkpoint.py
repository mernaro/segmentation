import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torchvision import transforms

class TeethDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        # Transformation par défaut : Redimensionner + Convertir en Tenseur (0-1)
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert("RGB") # On s'assure d'avoir 3 canaux
        mask = item['label'].convert("L")    # Masque en niveaux de gris (1 canal)
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image, mask

def prepare_dataloaders(hf_dataset, batch_size=4, train_ratio=0.8):
    # Création de l'objet Dataset global
    full_dataset = TeethDataset(hf_dataset)
    
    # Split Train / Validation
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Création des Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader