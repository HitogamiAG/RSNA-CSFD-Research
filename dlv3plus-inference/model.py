import segmentation_models_pytorch as smp
import torch
import os

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

def load_weights(model: torch.nn.Module) -> torch.nn.Module:
    """Load weights to model

    Args:
        model (torch.nn.Module): Model

    Returns:
        torch.nn.Module: Model with checkpoint weights
    """
    model.to(config.DEVICE)
    weights = torch.load(os.path.join(config.CHECKPOINT_PATH, 'best_model.pth'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(weights.state_dict())
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model