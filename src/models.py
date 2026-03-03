import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import os

def get_model(mode="vanilla", model_path=None, device="cpu"):
    if mode == "vanilla":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation='sigmoid'
        )
        print("🏗️ Modèle Vanilla (1 canal) initialisé.")

    elif mode == "pretrained":
        # 1. On crée d'abord le modèle standard en 3 canaux avec ImageNet
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        
       
        # Pour ResNet34, la première couche est model.encoder.conv1
        old_conv = model.encoder.conv1
        
        # On crée une nouvelle couche avec 1 seul canal d'entrée
        new_conv = nn.Conv2d(1, old_conv.out_channels, 
                             kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, 
                             padding=old_conv.padding, 
                             bias=old_conv.bias is not None)
        
        # On copie les poids : on fait la moyenne des 3 canaux ImageNet
        # Poids originaux : [64, 3, 7, 7] -> Nouveaux poids : [64, 1, 7, 7]
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias
        
        # On remplace l'ancienne couche par la nouvelle
        model.encoder.conv1 = new_conv
        
        print("🚀 Modèle Pretrained ADAPTÉ (1 canal + poids ImageNet moyennés) initialisé.")

    # 2. Chargement des poids (Inference)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Poids chargés depuis : {model_path}")

    return model.to(device)
