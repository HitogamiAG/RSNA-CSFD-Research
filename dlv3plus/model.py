import segmentation_models_pytorch as smp
import torch

import config

def initialize_model():
    """Initializes model

    Returns:
        Tuple[torch.nn.Module, callable]: Returns model & preprocessing function
    """
    model = smp.DeepLabV3Plus(
        encoder_name=config.ENCODER, 
        encoder_weights=config.ENCODER_WEIGHTS, 
        classes=len(config.CLASSES), 
        activation=config.ACTIVATION,
    )
    model = torch.nn.DataParallel(model)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)
    
    return model, preprocessing_fn