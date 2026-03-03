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
        
        # MODIFICATION ICI : "L" au lieu de "RGB" pour n'avoir qu'un canal (Gris)
        image = item['image'].convert("L") 
        mask = item['label'].convert("L") 
        
        # Le reste ne bouge pas
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image, mask

def prepare_dataloaders(hf_dataset, batch_size=4):
    full_dataset = TeethDataset(hf_dataset)
    total = len(full_dataset) # 116
    
    # On définit les tailles exactes
    val_size = 10
    test_size = 10
    train_size = total - val_size - test_size # 96
    
    # Split
    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Pour avoir toujours les mêmes images
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader