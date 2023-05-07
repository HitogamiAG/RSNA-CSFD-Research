import torch

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

from dataloader import RSNADataset

from typing import Dict

plt.ioff()

class Visualizer:
    """Visualize results during the training with logging to Wandb
    """
    def __init__(self, dataframe: pd.DataFrame, masks: Dict[str, np.array], confidence_threshold: float = 0.15):
        """Initialize Visualizer

        Args:
            dataframe (pd.DataFrame): Dataframe to visualize
            masks (Dict[str, np.array]): Preloaded segmentation masks
            confidence_threshold (float, optional): Visualize only results with confidence > trsh. Defaults to 0.15.
        """
        self.dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        self.confidence_threshold = confidence_threshold
        self.dataset = RSNADataset(dataframe = self.dataframe,
                                   masks =  masks,
                                   batch_size=1,
                                   device= wandb.config['device'],
                                   transform=None)
        # Generate random ids of records for dataframe
        self.ids_random = self.dataframe[(self.dataframe.iloc[:, 8:] != 0).any(axis=1)].sample(16).index
        # Constants with color & alpha-channel
        self.gt_color = [0, 255, 0, 128]
        self.pr_color = [0, 0, 255, 128]
    
    def visualize(self, model: torch.nn.Module):
        """Visualize results

        Args:
            model (torch.nn.Module): Trained model
        """
        i = 0
        
        model.eval()
        
        fig, axs = plt.subplots(4, 4, figsize = (20, 20))
        
        while i < 16:
            
            batch_index = self.ids_random[i]
            self.dataset.batch_index = batch_index

            X, y = self.dataset.__getitem__()
            if isinstance(X, type(None)):
                self.ids_random = self.dataframe[(self.dataframe.iloc[:, 8:] != 0).any(axis=1)].sample(1).index
                continue
            
            with torch.cuda.amp.autocast():
                preds = model(X)

            if len(preds) == 0:
                continue

            preds = preds[0]
            # RGBA image with alpha channel for masks
            rgba_img = np.zeros((X.shape[2], X.shape[3], 4))

            if len(y) == 0:
                continue

            y = y[0]

            ax = axs[i//4, i%4]

            # Orig image
            img = np.transpose((X.squeeze(dim = 0).cpu().numpy() * 255.0).astype(np.uint8), (1, 2, 0)).copy()
            
            # Scores
            scores = preds['scores'].detach().cpu().numpy().astype(np.float32)
            score_inds = np.where(scores > self.confidence_threshold)[0]
            
            # GT Bounding boxes
            gt_bounding_boxes = y['boxes'].cpu().numpy().astype(np.int32).tolist()
            pr_bounding_boxes = preds['boxes'].detach().cpu().numpy().astype(np.int32)[score_inds, :].tolist()

            for contour in gt_bounding_boxes:
                if len(contour) == 0:
                    continue
                cv.rectangle(img, (contour[0], contour[1]), (contour[2], contour[3]), self.gt_color[:3], 2)

            for contour in pr_bounding_boxes:
                if len(contour) == 0:
                    continue
                cv.rectangle(img, (contour[0], contour[1]), (contour[2], contour[3]), self.pr_color[:3], 2)

            # Get GT Masks
            gt_masks = y['masks'].sum(axis=0).cpu().numpy().astype(np.uint8)
            pr_masks = preds['masks'][score_inds, :, :, :].sum(axis=0).squeeze().detach().cpu().numpy().astype(np.uint8)

            # Convert numpy array of image into PIL Image object with alpha channel
            img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_RGB2RGBA))

            # Add to image GT masks
            gt_mask_pil = cv.cvtColor(gt_masks, cv.COLOR_GRAY2RGB)
            pr_mask_pil = cv.cvtColor(pr_masks, cv.COLOR_GRAY2RGB)

            gt_mask_pil = np.dstack([gt_mask_pil, gt_mask_pil[:, :, 0]])
            pr_mask_pil = np.dstack([pr_mask_pil, pr_mask_pil[:, :, 0]])

            gt_mask_pil = gt_mask_pil * np.array(self.gt_color)
            pr_mask_pil = pr_mask_pil * np.array(self.pr_color)

            gt_mask_pil = Image.fromarray(gt_mask_pil.astype(np.uint8))
            pr_mask_pil = Image.fromarray(pr_mask_pil.astype(np.uint8))

            img_pil.paste(gt_mask_pil, mask = gt_mask_pil)
            img_pil.paste(pr_mask_pil, mask = pr_mask_pil)

            # Plot image
            ax.imshow(img_pil)

            # Labels
            gt_labels = y['labels'].cpu().numpy().tolist()
            pr_labels = preds['labels'][score_inds].detach().cpu().numpy().tolist()

            #pred_labels = ...
            ax.set_title(f'{gt_labels}|{pr_labels if len(pr_labels) < 5 else str(pr_labels)[:16] + "..."}')

            i += 1
            
        wandb.log({"my_plot": fig})
        model.train()