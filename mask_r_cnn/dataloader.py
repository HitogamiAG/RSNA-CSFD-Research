import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision.transforms import functional as TF

import os
import pandas as pd
import numpy as np
from pathlib import Path

import cv2 as cv
import pydicom as dicom
import nibabel as nib
from sklearn.model_selection import train_test_split

from typing import Dict, Tuple
import wandb

import config


def check_dicom_orientation(meta_segm: pd.DataFrame):
    """Check the orientation of Z axis. If dict value is true
       then need to revert masks by Z axis.

    Args:
        meta_segm (pd.DataFrame): Dataframe with segmentation metadata

    Returns:
        Dict[str, bool]: Orientation results for each UID
    """
    orientation_check = {}

    for uid in meta_segm.StudyInstanceUID.unique():
        # Get metadata from DICOM files
        dcm1 = dicom.dcmread(os.path.join(
            config.DICOM_PATH, uid, (str(10) + '.dcm')))
        dcm2 = dicom.dcmread(os.path.join(
            config.DICOM_PATH, uid, (str(10) + '.dcm')))

        # Get Z coordinate and match for slice number 10 and 20
        if (dcm1.ImagePositionPatient[2] - dcm2.ImagePositionPatient[2]) > 0:
            orientation_check[uid] = True
        else:
            orientation_check[uid] = False
    return orientation_check


def load_segmentation_masks(meta_segm: pd.DataFrame, orientation_check: Dict[str, bool]):
    """Load segmentation masks in RAM to speed up DataLoader

    Args:
        meta_segm (pd.DataFrame): Dataframe with segmentation metadata
        orientation_check (Dict[str, bool]): Orientation results for each UID
    Returns:
        Dict[str, np.array]: Segmenatation masks for each UID
    """
    masks = {}
    for uid in meta_segm.StudyInstanceUID.unique():
        # Load mask
        mask = nib.load(os.path.join(config.SEGMENTATION_PATH, (uid + '.nii')))
        mask = np.asarray(mask.get_data())
        # Revert by Z coord if needed
        if orientation_check[uid]:
            mask = mask[:, :, ::-1]
        mask = np.rot90(mask, k=1, axes=(0, 1))
        masks[uid] = mask
    return masks


def split_uids(meta_segm: pd.DataFrame):
    """Split dataset by UIDs into train/val/test

    Args:
        meta_segm (pd.DataFrame): Dataframe with segmentation metadata

    Returns:
        Tuple[List[str]]: Tuple of lists of UIDs
    """
    train_UIDs, test_val_UIDs = train_test_split(meta_segm.StudyInstanceUID.unique(),
                                                 test_size=wandb.config['val_size'] +
                                                 wandb.config['test_size'],
                                                 random_state=42)
    val_UIDs, test_UIDs = train_test_split(test_val_UIDs,
                                           test_size=wandb.config['test_size'] / (wandb.config['test_size'] \
                                               + wandb.config['val_size']),
                                           random_state=42)

    print(f'Number of UIDs in train: {len(train_UIDs)}')
    print(f'Number of UIDs in val: {len(val_UIDs)}')
    print(f'Number of UIDs in test: {len(test_UIDs)}')

    train_UIDs, val_UIDs, test_UIDs = train_UIDs.tolist(
    ), val_UIDs.tolist(), test_UIDs.tolist()
    return train_UIDs, val_UIDs, test_UIDs


def load_dataframes_and_masks():
    """Generate train/val/test dataframes and load masks

    Returns:
        Tuple[pd.DataFrame, Dict[str, np.array]]: Returns train/val/test dataframes and masks
    """
    # Check available segmentation masks
    available_uids = [uid.replace('.nii', '')
                      for uid in os.listdir(config.SEGMENTATION_PATH)]

    # Load and filter metadata
    meta_segm = pd.read_csv(os.path.join(
        config.METADATA_PATH, 'meta_segmentation_clean.csv'))
    meta_segm = meta_segm[meta_segm.StudyInstanceUID.isin(available_uids)]

    # Remove any records where there is no vertebra for segment
    meta_segm = meta_segm[(meta_segm.iloc[:, 8:] != 0).any(1)]

    # Load masks
    orientation_check = check_dicom_orientation(meta_segm)
    masks = load_segmentation_masks(meta_segm, orientation_check)

    # Load dataframes
    train_UIDs, val_UIDs, test_UIDs = split_uids(meta_segm)
    train_df = meta_segm[meta_segm.StudyInstanceUID.isin(train_UIDs)]
    val_df = meta_segm[meta_segm.StudyInstanceUID.isin(val_UIDs)]
    test_df = meta_segm[meta_segm.StudyInstanceUID.isin(test_UIDs)]

    print(f'Shape of train dataframe: {train_df.shape}')
    print(f'Shape of val dataframe: {val_df.shape}')
    print(f'Shape of test dataframe: {test_df.shape}')

    return (train_df, val_df, test_df, masks)


class RSNADataset(Dataset):
    """Custom PyTorch dataset.
       Note: The data format for Mask R-CNN is Tuple[torch.Tensor, Dict[str, torch.Tensor]]
             Dataloader is not able to stack dictionaries so in this dataset
             was realized functionality to iterate over this dataset.
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 masks: Dict[str, np.array],
                 dicom_path: str = config.DICOM_PATH,
                 segm_path: str = config.SEGMENTATION_PATH,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 device: torch.device = torch.device('cpu'),
                 transform: Dict[str, int] =None) -> None:
        """Initialization of dataset

        Args:
            dataframe (pd.Dataframe): Train/val/test dataframe
            dicom_path (str): Path to DICOM folders
            segm_path (str): Paht to segmentation folder
            batch_size (int): Batch size
            num_workers (int): Load in parallel
            device (torch.device): Device
            masks (Dict[str, np.array]): Preloaded segmentation masks
            transform (Dict[str, float], optional): Dict with augmentation params & probs. Defaults to None.
        """
        super(RSNADataset, self).__init__()
        self.dataframe = dataframe
        self.dicom_path = dicom_path
        self.segm_path = segm_path
        self.masks = masks

        self.batch_index = -1
        
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.transform = transform

    def load_dicom(self, path):
        """Load DICOM (.dcm) file

        Args:
            path (str): Path to DICOM file

        Returns:
            Tuple[np.array, Dict[str, Any]]: Returns image & metadata
        """
        # Source: https://www.kaggle.com/code/vslaykovsky/pytorch-effnetv2-vertebrae-detection-acc-0-95
        img = dicom.dcmread(path)
        img.PhotometricInterpretation = 'YBR_FULL'
        data = img.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return cv.cvtColor(data, cv.COLOR_GRAY2RGB), img

    def generate_data_batch(self, ind):
        """Get batch by index

        Args:
            ind (int): Index of batch

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Returns image tensor &
                dictionary with class, bounding box & segmentation mask
        """

        # Targets - y
        # Targets is array of dictionaries. Each dictionary contains masks, bounding boxes and labels for 1 record of patient data
        self.targets = []

        # Imgs - X
        # Imgs contains DICOM images. The shape is (batch_size, 3, 512, 512)
        # np.array([], dtype=np.uint8).reshape(0, 3, 512, 512)
        self.imgs = torch.empty((0, 3, 512, 512), dtype=torch.float32)

        # Get records from dataframe about what data should be loaded in the current batch
        batches = self.dataframe.iloc[ind *
                                      self.batch_size:(ind+1)*self.batch_size, :].to_numpy()

        # Multithreading for loading data

        for batch in batches:
            self.extract_data_from_batch(batch)

        # Convert images to tensors
        # self.imgs = torch.as_tensor(self.imgs, dtype=torch.float32)

        if self.imgs.shape[0] == 0:
            print('no imgs')
            return self.__getitem__()

        # Return batch data, e.g. X and y
        return self.imgs.to(self.device, non_blocking=True), self.targets

    def transform_function(self, img, mask):
        """Apply transformation to image and mask

        Args:
            img (np.array): Image
            mask (np.array): Mask

        Returns:
            Tuple[np.array]: Returns augmented image and mask
        """
        transform_config = self.transform

        img, mask = TF.to_pil_image(
            img), TF.to_pil_image(mask.astype(np.uint8))

        if np.random.random() < transform_config['p_original']:
            pass
        else:
            # Horizontal flip
            if transform_config['p_hflip'] < np.random.random():
                img, mask = TF.hflip(img), TF.hflip(mask)

            # Affine
            if transform_config['p_affine'] < np.random.random():
                affine_params = tv.transforms.RandomAffine(30).get_params(
                    (-15, 15), (0.1, 0.1), (1, 1), (-15, 15), (512, 512))
                img = TF.affine(img, *affine_params)
                mask = TF.affine(mask, *affine_params)

            if transform_config['p_cjitter'] < np.random.random():
                img = tv.transforms.ColorJitter(*(0.3, 0.3, 0.1, 0.1))(img)

        return np.asarray(img), np.asarray(mask)

    def extract_data_from_batch(self, batch: pd.Series):
        """Get batch by record from dataframe

        Args:
            batch (pd.Series): Record from dataframe
        """
        # Define initial structure of y
        target = {
            'boxes': [],
            'labels': None,
            'masks': []
        }

        # Get mask from array of masks
        mask = self.masks[batch[0]][:, :, batch[1] - 1]

        # Get vertebrae numbers then is presented on mask. C1 - 1, C2 - 2 ...
        labels = np.unique(mask)[1:]

        labels_to_dict = []

        # if there is no vertebrae on mask
        if len(labels) < 2:
            return None
        
        # Load Dicom image
        img = self.load_dicom(
            os.path.join(self.dicom_path, batch[0], (str(batch[1]) + '.dcm')))[0]

        # Apply transformations
        if self.transform:
            img, mask = self.transform_function(img, mask)

        for label in labels:
            label_mask = (mask == label).astype(np.uint8)

            # Here if we have some parts of vertebra on the mask are placed separately, they should be processed as different object (due to architecture of Mask R-CNN)
#             segmented_mask = measure.label(label_mask)

#             for e_object in np.unique(segmented_mask)[1:]:
#                 object_mask = (segmented_mask == e_object).astype(np.uint8)

            x, y, w, h = cv.boundingRect(label_mask)

            if sum([x, y, w, h]) == 0:
                print('cv.boundingRect not found mask')
                continue
            elif w < 5 and h < 5:
                continue

            if label.item() < 8:
                labels_to_dict.append(label.item())
            else:
                # 0 is the class for other vertebraes that can be faced on the mask with C1-C7
                labels_to_dict.append(0)
            target['masks'].append(label_mask)
            target['boxes'].append(np.array([x, y, x+w, y+h]))

        # Convert to tensor
        target['labels'] = torch.from_numpy(np.array(labels_to_dict).astype(
            np.int64)).to(self.device, non_blocking=True)
        target['boxes'] = torch.from_numpy(np.array(target['boxes']).astype(
            np.float32)).to(self.device, non_blocking=True)
        target['masks'] = torch.from_numpy(np.array(target['masks']).astype(
            np.float32)).to(self.device, non_blocking=True)

        img = tv.transforms.ToTensor()(img).unsqueeze(dim=0)

        self.imgs = torch.cat([self.imgs, img], dim=0)
        self.targets.append(target)

    def __len__(self):
        """Return number of batches

        Returns:
            int: Number of batches
        """
        return int(len(self.dataframe) / self.batch_size)

    # the internal counter of batches
    def reset_batch(self):
        """Reset batch counters and shuffle dataframe
        """
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        self.batch_index = -1

    # Check are there batches or it's end of dataframe and we have to do next iteration
    def is_end(self):
        """Return True if there are no batches

        Returns:
            bool: result
        """
        return False if self.batch_index < self.__len__() - 1 else True

    def __getitem__(self) -> None:
        """Get item.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Returns image tensor &
                dictionary with class, bounding box & segmentation mask
        """
        # If there is no batches
        if self.is_end():
            return None, None

        self.batch_index += 1
        return self.generate_data_batch(self.batch_index)
