import torch
import numpy as np
import wandb

import config

def set_seeds():
    """Set seeds to random generators
    """
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def checkpoint(model: torch.nn.Module,
               checkpoint_path: str = config.CHECKPOINT_PATH,
               model_name:str = 'checkpoint'):
    """Save model state and log to Wandb

    Args:
        model (torch.nn.Module): Model
        checkpoint_path (str): Path to checkpoints folder
        model_name (str): Checkpoint name
    """
    if not checkpoint_path.is_dir():
        checkpoint_path.mkdir()
    
    # Save trained model weights
    save_path = checkpoint_path / (model_name + '.pth')
    torch.save(model.state_dict(), save_path)

    # Upload them on wandb
    artifact = wandb.Artifact(model_name, type='checkpoint')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    print(f'Logged {model_name}')