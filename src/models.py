import segmentation_models_pytorch as smp

def get_model(mode="segmentation"):
    if mode == "segmentation":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None # On utilisera Sigmoid ou la Loss gérera ça
        )
    return model