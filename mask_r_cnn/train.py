import argparse
import torch
import numpy as np
import pandas as pd
import os
import wandb

import config
from dataloader import load_dataframes_and_masks, RSNADataset
from model import initialize_model
from visualizer import Visualizer
from logger import initiate_wandb_run, Logger
from utils import set_seeds, checkpoint

from typing import Dict, Tuple

def train(
    model: torch.nn.Module,
    masks: Dict[str, np.array],
    dfs: Tuple[pd.DataFrame],
    dls: Tuple[torch.utils.data.Dataset],
    loggers: Tuple[Logger],
    visualizer: Visualizer
):
    """Train model

    Args:
        model (torch.nn.Module): Mask R-CNN Model
        masks (Dict[str, np.array]): Preloaded segmentation masks
        dfs (Tuple[pd.DataFrame]): Dataframes
        dls (Tuple[torch.utils.data.Dataset]): Dataloaders
        loggers (Tuple[Logger]): Loggers
        visualizer (Visualizer): Visualizer
    """
    train_df, val_df = dfs
    train_dataset, val_dataset = dls
    train_logger, val_logger = loggers
    
    # Configuration
    max_num_of_iters = int(wandb.config['n_epochs'] * train_dataset.dataframe.shape[0] / wandb.config['batch_size'])
    train_send_logs_each_n_iters = 5
    
    validate_n_times_per_epoch = 5
    validate_each_n_iters = round(train_dataset.dataframe.shape[0] / train_dataset.batch_size / validate_n_times_per_epoch)
    
    vizualize_n_times_per_epoch = 3
    vizualize_each_n_iters = round(train_dataset.dataframe.shape[0] / train_dataset.batch_size / vizualize_n_times_per_epoch)
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.002, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_num_of_iters, eta_min=0.0002)
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    model.to(wandb.config['device'])
    
    print('Train stage configured. Starting train...')
    
    for i in range(wandb.config['n_epochs']):
        wandb.log({'epoch' : i+1})
        train_dataset.reset_batch()
        while not train_dataset.is_end():
            X, y = train_dataset.__getitem__()
            if isinstance(X, type(None)): break;
            
            # TODO: call func to get predictions and losses at once
            with torch.cuda.amp.autocast():
                loss_dict = model(X, y)
            
            optimizer.zero_grad()
            losses = sum(loss for loss in loss_dict.values()).sum()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            #warmup.step()
            # TODO: Calculate APs and mAP
            # Log iteration results
            train_logger.log({
                'loss_classifier' : loss_dict['loss_classifier'].sum().item(),
                'loss_box_reg' : loss_dict['loss_box_reg'].sum().item(),
                'loss_mask' : loss_dict['loss_mask'].sum().item(),
                'loss_objectness' : loss_dict['loss_objectness'].sum().item(),
                'loss_rpn_box_reg' : loss_dict['loss_rpn_box_reg'].sum().item(),
                'total_loss' : losses.sum().item()
            })
            
            if train_logger.update_counter == train_send_logs_each_n_iters:
                wandb.log({'lr' : optimizer.param_groups[0]['lr']})
                train_logger.send_logs()
            
            # Visualize model each N iterations
            if (train_dataset.batch_index + 1) % vizualize_each_n_iters == 0:
                visualizer.visualize(model)
            
            # Validate model each N iterations
            if (train_dataset.batch_index + 1) % validate_each_n_iters == 0:
                    
                val_dataset.reset_batch()
                while not val_dataset.is_end():
                    X, y = val_dataset.__getitem__()
                    if isinstance(X, type(None)): break;
                    
                    # TODO: call func to get predictions and losses at once
                    with torch.cuda.amp.autocast():
                        loss_dict = model(X, y)
                    
                    losses = sum(loss for loss in loss_dict.values()).sum()

                    # TODO: Calculate APs and mAP
                    # Log validation results
                    val_logger.log({
                        'loss_classifier' : loss_dict['loss_classifier'].sum().item(),
                        'loss_box_reg' : loss_dict['loss_box_reg'].sum().item(),
                        'loss_mask' : loss_dict['loss_mask'].sum().item(),
                        'loss_objectness' : loss_dict['loss_objectness'].sum().item(),
                        'loss_rpn_box_reg' : loss_dict['loss_rpn_box_reg'].sum().item(),
                        'total_loss' : losses.sum().item()
                    })
                val_logger.send_logs()
        
        train_logger.send_logs()
        visualizer.visualize(model)
        
        checkpoint(model, config.CHECKPOINT_PATH, f'chkp_mrcnn_{i+1}')
        
    wandb.finish()
    
    
if __name__ == '__main__':
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
    
    # Initiate wandb
    
    initiate_wandb_run(wb_config)
    print(f'Device: {wb_config["device"]}')
    
    set_seeds()
    
    # Initiate dataframes
    
    train_df, val_df, _, masks = load_dataframes_and_masks()
    print('Masks & Dataframes logged')
    
    # Shuffle dataframes
    
    train_df = train_df.sample(frac=1).reset_index(drop = True)
    val_df = val_df.sample(frac=1).reset_index(drop = True)
    
    # Initiate datasets
    
    train_dataset = RSNADataset(dataframe = train_df,
                                masks = masks,
                                device= wandb.config['device'],
                                transform = config.TRANSFORM)
    val_dataset = RSNADataset(dataframe = val_df,
                              masks = masks,
                              device= wandb.config['device'],
                              transform = config.TRANSFORM)
    print('Datasets loaded')
    
    # Initiate loggers
    
    train_logger = Logger(prefix='train_', metrics_to_log={
            'loss_classifier' : 0,
            'loss_box_reg' : 0,
            'loss_mask' : 0,
            'loss_objectness' : 0,
            'loss_rpn_box_reg' : 0,
            'total_loss' : 0
        })
    
    val_logger = Logger(prefix='val_', metrics_to_log={
            'loss_classifier' : 0,
            'loss_box_reg' : 0,
            'loss_mask' : 0,
            'loss_objectness' : 0,
            'loss_rpn_box_reg' : 0,
            'total_loss' : 0
        })
    print('Loggers initialized')
    
    # Initiate visualizer
    
    visualizer = Visualizer(val_df, masks)
    print('Visualizer initialized')
    
    # Initiate model
    
    model = initialize_model()
    print('Model initialized')
    
    # Start train
    
    train(model,
          masks,
          (train_df, val_df),
          (train_dataset, val_dataset),
          (train_logger, val_logger),
          visualizer
          )