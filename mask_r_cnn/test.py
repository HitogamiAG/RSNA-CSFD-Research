import torch
import argparse
import os
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

from model import initialize_model
from utils import set_seeds, convert_mask, get_pixel_accuracy, get_iou, get_dc
from logger import initiate_config_only
from dataloader import load_dataframes_and_masks, RSNADataset
import config

from typing import Dict
import warnings
warnings.filterwarnings('ignore')

def test(
    model: torch.nn.Module,
    test_dataset: RSNADataset
):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    model.to(wandb.config['device'])
    
###########################
####### Test loss #########
###########################
    
    losses = {
        'loss_classifier' : 0,
        'loss_box_reg' : 0,
        'loss_mask' : 0,
        'loss_objectness' : 0,
        'loss_rpn_box_reg' : 0
    }
    counter = 0
    
    test_dataset.reset_batch()
    print(f'Starting loss test...')
    pbar = tqdm()
    
    while not test_dataset.is_end():
        X, y = test_dataset.__getitem__()
        if isinstance(X, type(None)): break;

        loss_dict: dict = model(X, y)
        
        for key, value in loss_dict.items():
            losses[key] += value.sum().item()
        
        counter += 1
        pbar.update(1)
        
    print('-'*30)
    print('Loss measure results:')
    for key, value in losses.items():
        losses[key] = value / counter
        print(f'{key} : {value / counter}')
    losses['total_loss'] = sum(losses.values())
    print(f'total_loss : {losses["total_loss"]}')
    
    del X, y
    torch.cuda.empty_cache()
    gc.collect()
    
###########################
###### Metrics test #######
###########################
    
    model.eval()
    test_dataset.reset_batch()
    
    pixel_acc = {}
    iou = {}
    dc = {}
    print(f'Starting metrics test...')
    pbar = tqdm()
    
    while not test_dataset.is_end():
        X, y = test_dataset.__getitem__()
        if isinstance(X, type(None)): break;
        y = y[0]
        
        # List of dicts of boxes, labels, scores, masks
        preds = model(X)[0]
        
        pr_mask = convert_mask(
            preds['labels'].detach().cpu().numpy(),
            (preds['masks'].detach().cpu().numpy() > 0.5).astype(np.bool8)
        )
        
        gt_mask = convert_mask(
            y['labels'].cpu().numpy() + 1,
            y['masks'].unsqueeze(dim=1).cpu().numpy().astype(np.bool8)
        )
        
        # Pixel accuracy
        pixel_acc_instance = get_pixel_accuracy(pr_mask, gt_mask)
        for k, v in pixel_acc_instance.items():
            if not k in pixel_acc:
                pixel_acc[k] = {'correct' : [],
                                'total' : []}
            pixel_acc[k]['correct'].append(v[0])
            pixel_acc[k]['total'].append(v[1])
        
        # Intersection over Union
        iou_instance = get_iou(pr_mask, gt_mask)
        for k, v in iou_instance.items():
            if not k in iou:
                iou[k] = {
                    'intersection' : [],
                    'union' : []}
            iou[k]['intersection'].append(v[0])
            iou[k]['union'].append(v[1])
            
        # Dice coefficient
        dc_instance = get_dc(pr_mask, gt_mask)
        for k, v in dc_instance.items():
            if not k in dc:
                dc[k] = {
                    'intersection' : [],
                    'union' : []
                }
            dc[k]['intersection'].append(v[0])
            dc[k]['union'].append(v[1])
            
        pbar.update(1)
        
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='Number of workers for data loading')
    parser.add_argument('--train_size', type=float, default=0.34, help='Training set size')
    parser.add_argument('--val_size', type=float, default=0.33, help='Validation set size')
    parser.add_argument('--test_size', type=float, default=0.33, help='Test set size')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to train on')

    args = parser.parse_args()
    
    # Initiate config
    
    wb_config = {
        'batch_size' : args.batch_size,
        'num_workers' : args.num_workers,
        # Only for code test. Training ratio: 0.935/0.033/0.032
        'train_size' : args.train_size,
        'val_size' : args.val_size,
        'test_size' : args.test_size,
        'device' : args.device,
    }
    
    initiate_config_only(wb_config)
    
    # Initialize data
    
    _, _, test_df, masks = load_dataframes_and_masks()
    print('Masks & Dataframes loaded')
    
    test_dataset = RSNADataset(dataframe = test_df,
                              masks = masks,
                              device= wandb.config['device'],
                              transform = config.TRANSFORM)
    print('Datasets loaded')
    
    # Initialize model
    
    model = initialize_model()
    print('Model initialized')

    # Load model
    weights = torch.load(os.path.join(config.CHECKPOINT_PATH, 'chkp_mrcnn_10.pth'))
    model.load_state_dict(
        {key.replace('module.', '') : value for key, value in weights.items()}
    )
    
    print('Model weights loaded')
    
    # Run tests
    
    test(
        model,
        test_dataset
    )