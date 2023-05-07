import argparse
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils as smp_utils
import os
import numpy as np
import pandas as pd
import wandb

from utils import set_seeds
import config
from logger import initiate_wandb_run
from dataloader import (
    load_dataframes_and_masks,
    shuffle_train_df,
    BuildingDataset,
    get_preprocessing
)
from patches import _format_logs, TrainEpoch
from model import initialize_model

from typing import Tuple, Dict, Callable

def warmup_lambda(current_step):
    """Lambda function for LR scheduler

    Args:
        current_step (float): Epoch number with floats

    Returns:
        float: Learning rate value
    """
    if current_step < 0.5:
        return current_step / 1000
    elif current_step < 1:
        return 0.0005
    elif current_step < 6:
        return 0.0001
    else:
        return 0.00005

def train(model: torch.nn.Module,
          preprocessing_fn: Callable,
          dfs: Tuple[pd.DataFrame, pd.DataFrame],
          masks: Dict[str, np.array],
          dls: Tuple[DataLoader]):
    """Training function

    Args:
        model (torch.nn.Module): Model
        preprocessing_fn (Callable): Preprocessing function
        dfs (Tuple[pd.DataFrame]): Dataframes
        masks (Dict[str, np.array]): Preloaded segmentation masks
        dls (Tuple[DataLoader]): Dataloaders
    """
    train_df, val_df = dfs
    train_loader, val_loader = dls
    # define loss function
    loss = smp_utils.losses.DiceLoss()

    # define metrics
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]

    # define optimizer
    optimizer = torch.optim.AdamW([
        dict(params=model.parameters(), lr=1, weight_decay = 5e-5),
    ])
    
    # define scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    train_epoch = smp_utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        loader_size = len(train_loader),
        scheduler = lr_scheduler,
        device=wandb.config['device'],
        verbose=True
    )

    valid_epoch = smp_utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=wandb.config['device'],
        verbose=True,
    )
    
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, config.EPOCHS):
        wandb.log({
            'epoch' : i+1
        })
        
        # Restart train loader
        train_dataset = BuildingDataset(dataframe = shuffle_train_df(train_df),
                                        masks = masks,
                                        preprocessing = get_preprocessing(preprocessing_fn))
        train_loader = DataLoader(train_dataset,
                              batch_size = wandb.config['batch_size'],
                              shuffle = True,
                              num_workers = wandb.config['num_workers'],
                              drop_last = True)
        
        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, os.path.join(config.CHECKPOINT_PATH, 'best_model.pth'))
            print('Model saved!')
    
    model_name = 'dlv3plus'
    save_path = './best_model.pth'
    artifact = wandb.Artifact(model_name, type='checkpoint')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    print(f'Logged {model_name}')
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='Number of workers for data loading')
    parser.add_argument('--train_size', type=float, default=0.34, help='Training set size')
    parser.add_argument('--val_size', type=float, default=0.33, help='Validation set size')
    parser.add_argument('--test_size', type=float, default=0.33, help='Test set size')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to train on')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train for')

    args = parser.parse_args()
    
    wb_config = {
        'batch_size' : args.batch_size,
        'val_batch_size' : args.val_batch_size,
        'num_workers' : args.num_workers,
        # Only for code test. Training ratio: 0.935/0.033/0.032
        'train_size' : args.train_size,
        'val_size' : args.val_size,
        'test_size' : args.test_size,
        'device' : args.device,
        'n_epochs' : args.n_epochs
    }
    
    # Set seeds
    
    set_seeds()
    
    # Initialize wandb
    
    initiate_wandb_run(wb_config)
    
    # Prepare dataframes
    
    train_df, val_df, _, masks = load_dataframes_and_masks()
    
    # Initialize model
    
    model, preprocessing_fn = initialize_model()
    
    # Initialize dataloaders
    
    train_dataset = BuildingDataset(
        dataframe= shuffle_train_df(train_df),
        masks = masks,
        preprocessing = get_preprocessing(preprocessing_fn)
    )
    val_dataset = BuildingDataset(
        dataframe= shuffle_train_df(val_df),
        masks = masks,
        preprocessing = get_preprocessing(preprocessing_fn)
    )
    
    train_loader = DataLoader(train_dataset,
                              batch_size = wandb.config['batch_size'],
                              shuffle=True,
                              num_workers=wandb.config['num_workers'],
                              drop_last=True
                              )
    val_loader = DataLoader(val_dataset,
                              batch_size = wandb.config['val_batch_size'],
                              shuffle=False,
                              num_workers=wandb.config['num_workers'],
                              drop_last=True
                              )
    
    # Patch SMP
    
    smp_utils.train.Epoch._format_logs = _format_logs
    smp_utils.train.TrainEpoch = TrainEpoch
    
    # Start training
    
    train(
        model = model,
        preprocessing_fn = preprocessing_fn,
        dfs = (train_df, val_df),
        masks = masks,
        dls = (train_loader, val_loader)
    )