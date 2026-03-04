import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import os

def get_model(mode="vanilla", model_path=None, device="cpu"):
    """
    Modes disponibles :
    - vanilla : 1 canal, poids aléatoires, sigmoid.
    - pretrained : 1 canal, poids ImageNet moyennés, sigmoid.
    - reconstruction : 1 canal, poids ImageNet, SANS sigmoid (Phase 1).
    - segmentation : identique à pretrained (Phase 2).
    """
    
    # 1. Création de la structure de base (on commence par 3 canaux pour ImageNet)
    # Si mode est vanilla, on ne charge pas de poids.
    weights = "imagenet" if mode != "vanilla" else None
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation=None # On gère l'activation manuellement plus bas
    )

    # 2. Adaptation systématique de la couche d'entrée (3 canaux -> 1 canal)
    # Cela permet d'utiliser le data_loader en mode "L" (Gris) pour tous les modes
    old_conv = model.encoder.conv1
    new_conv = nn.Conv2d(1, old_conv.out_channels, 
                         kernel_size=old_conv.kernel_size, 
                         stride=old_conv.stride, 
                         padding=old_conv.padding, 
                         bias=old_conv.bias is not None)
    
    with torch.no_grad():
        if weights == "imagenet":
            new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias
    
    model.encoder.conv1 = new_conv

    # 3. Configuration spécifique selon le mode
    if mode == "reconstruction":
        print(" Mode RECONSTRUCTION (Phase 1) : Pas de Sigmoid.")
        model.segmentation_head[2] = nn.Identity() # Sortie linéaire pour les pixels
    else:
        # Pour vanilla, pretrained et segmentation, on veut la Sigmoid
        print(f" Mode {mode.upper()} : Activation Sigmoid activée.")
        model.segmentation_head[2] = nn.Sigmoid()

    # 4. Chargement des poids si un chemin est fourni (Inference)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f" Poids chargés depuis : {model_path}")

    return model.to(device)