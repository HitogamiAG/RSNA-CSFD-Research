import argparse
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils as smp_utils
import os
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from utils import (
    set_seeds,
    convert_mask,
    get_pixel_accuracy,
    get_dc,
    get_iou
)
import config
from logger import initiate_config_only
from dataloader import (
    load_dataframes_and_masks,
    shuffle_train_df,
    BuildingDataset,
    get_preprocessing
)
from patches import _format_logs, TrainEpoch
from model import initialize_model

from typing import Tuple, Dict, Callable

def test(
    model: torch.nn.Module,
    test_loader: BuildingDataset
):
    # model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False
        
    loss_fn = smp_utils.losses.DiceLoss()
    
    test_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss_fn,
        metrics=[
            smp_utils.metrics.IoU(threshold=0.5),
        ],
        device=wandb.config['device'],
        verbose=True
    )
    
    test_epoch.run(test_loader)
    
    model.eval()
    pixel_acc = {}
    iou = {}
    dc = {}

    for X, y in tqdm(test_loader):
        X, y = X.to(wandb.config['device']), y.to(wandb.config['device'])
        preds = model(X)
        
        preds = preds.squeeze(dim=0).detach().cpu().numpy()
        y = y.squeeze(dim=0).cpu().numpy()
            
        preds = np.argmax(preds, axis=0)
        y = np.argmax(y, axis=0)
        
        pixel_acc_instance = get_pixel_accuracy(preds, y)
        for k, v in pixel_acc_instance.items():
            if not k in pixel_acc:
                pixel_acc[k] = {'correct' : [],
                                'total' : []}
            pixel_acc[k]['correct'].append(v[0])
            pixel_acc[k]['total'].append(v[1])
        
        iou_instance = get_iou(preds, y)
        for k, v in iou_instance.items():
            if not k in iou:
                iou[k] = {
                    'intersection' : [],
                    'union' : []}
            iou[k]['intersection'].append(v[0])
            iou[k]['union'].append(v[1])
        
        dc_instance = get_dc(preds, y)
        for k, v in dc_instance.items():
            if not k in dc:
                dc[k] = {
                    'intersection' : [],
                    'union' : []
                }
            dc[k]['intersection'].append(v[0])
            dc[k]['union'].append(v[1])

    print(f'Pixel accuracy:')
    for k, v in pixel_acc.items():
        pixel_acc[k] = np.sum(v['correct']) / np.sum(v['total'])
        print(f'{k} : {pixel_acc[k]}')

    print(f'Intersection over Union:')
    for k, v in iou.items():
        iou[k] = np.sum(v['intersection']) / np.sum(v['union'])
        print(f'{k} : {iou[k]}')
    print(f'Mean: {np.mean([v for v in iou.values()])}')

    print(f'Dice coefficient:')
    for k, v in dc.items():
        dc[k] = np.sum(v['intersection']) / np.sum(v['union'])
        print(f'{k} : {dc[k]}')
    print(f'Mean: {np.mean([v for v in dc.values()])}')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='Number of workers for data loading')
    parser.add_argument('--train_size', type=float, default=0.34, help='Training set size')
    parser.add_argument('--val_size', type=float, default=0.33, help='Validation set size')
    parser.add_argument('--test_size', type=float, default=0.33, help='Test set size')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to train on')

    args = parser.parse_args()
    
    wb_config = {
        'batch_size' : args.batch_size,
        'num_workers' : args.num_workers,
        # Only for code test. Training ratio: 0.935/0.033/0.032
        'train_size' : args.train_size,
        'val_size' : args.val_size,
        'test_size' : args.test_size,
        'device' : args.device
    }
    
    # Set seeds
    
    set_seeds()
    
    # Initialize wandb
    
    initiate_config_only(wb_config)
    
    # Prepare dataframes
    
    _, _, test_df, masks = load_dataframes_and_masks()
    
    # Initialize model
    
    model, preprocessing_fn = initialize_model()
    
    # Load weights
    model.load_state_dict(
        torch.load(
            os.path.join(config.CHECKPOINT_PATH, 'best_model.pth')
        ).state_dict()
    )
    
    # Initialize dataloaders
    
    test_dataset = BuildingDataset(
        dataframe= shuffle_train_df(test_df),
        masks = masks,
        preprocessing = get_preprocessing(preprocessing_fn)
    )
    
    test_loader = DataLoader(test_dataset,
                              batch_size = wandb.config['batch_size'],
                              shuffle=True,
                              num_workers=wandb.config['num_workers'],
                              drop_last=True
                              )
    
    # Start training
    
    test(
        model,
        test_loader
    )