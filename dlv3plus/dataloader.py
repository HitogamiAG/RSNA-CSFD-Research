import torch
from torch.utils.data import Dataset, DataLoader
import pydicom as dicom
import nibabel as nib
import albumentations as album
import os
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

import config

from typing import Dict

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

    # Load masks
    orientation_check = check_dicom_orientation(meta_segm)
    masks = load_segmentation_masks(meta_segm, orientation_check)

    # Load dataframes
    train_UIDs, val_UIDs, test_UIDs = split_uids(meta_segm)
    test_UIDs = [
        '1.2.826.0.1.3680043.5783',
        '1.2.826.0.1.3680043.10633',
        '1.2.826.0.1.3680043.32658'
    ]
    train_df = meta_segm[meta_segm.StudyInstanceUID.isin(train_UIDs)]
    val_df = meta_segm[meta_segm.StudyInstanceUID.isin(val_UIDs)]
    test_df = meta_segm[meta_segm.StudyInstanceUID.isin(test_UIDs)]

    print(f'Shape of train dataframe: {train_df.shape}')
    print(f'Shape of val dataframe: {val_df.shape}')
    print(f'Shape of test dataframe: {test_df.shape}')

    return (train_df, val_df, test_df, masks)

def shuffle_train_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Shuffle train dataframe to extrain all slices with vertrebra + n% of non-relevant slices

    Args:
        train_df (pd.DataFrame): Train dataframe

    Returns:
        pd.DataFrame: Shuffled and filtered dataframe
    """
    # Get first and last occurancies of relative slices (C1-C7)
    minmax_traindf = train_df[(train_df.iloc[:, 8:] != 0).any(1)].groupby('StudyInstanceUID').agg({'Slice':['min', 'max']})

    # Label slices: 1 if slice before first occurance | 2 if relative slice | 3 otherwise
    mask_train_df = train_df[['StudyInstanceUID', 'Slice']]
    mask_train_df['pos_label'] = mask_train_df.apply(lambda x: 1 if x.Slice < minmax_traindf.loc[x.StudyInstanceUID].min() \
                                                    else 3 if x.Slice > minmax_traindf.loc[x.StudyInstanceUID].max()\
                                                    else 2, axis=1)
    
    records_to_get = pd.concat([
        mask_train_df[mask_train_df.pos_label == 2],
        mask_train_df[mask_train_df.pos_label == 1].sample(frac=0.1, random_state=42),
        mask_train_df[mask_train_df.pos_label == 3].sample(frac=0.1, random_state=42)
    ])
    
    return pd.merge(train_df, records_to_get[["StudyInstanceUID", "Slice"]], on=["StudyInstanceUID", "Slice"], how="inner")

class BuildingDataset(Dataset):
    """Custom dataset for segmentation task
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 masks: Dict[str, np.array],
                 dicom_path: str = config.DICOM_PATH,
                 augmentation: album.Compose = None,
                 preprocessing: album.Compose = None):
        """Initializes dataset

        Args:
            dataframe (pd.DataFrame): Dataframe
            masks (Dict[str, np.array]): Preloaded segmentation masks
            dicom_path (str, optional): Path to DICOM files. Defaults to config.DICOM_PATH.
            augmentation (album.Compose, optional): Augmentation function. Defaults to None.
            preprocessing (album.Compose, optional): Preprocessing function. Defaults to None.
        """
        super(BuildingDataset, self).__init__()
        self.class_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.dataframe = dataframe
        self.dicom_path = dicom_path
        self.masks = masks
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __len__(self):
        """Get dataset length

        Returns:
            int: Dataset length
        """
        return len(self.dataframe)
    
    def load_dicom(self, path):
        """Load DICOM (.dcm) file

        Args:
            path (str): Path to DICOM file

        Returns:
            Tuple[np.array, Dict[str, Any]]: Returns image & metadata
        """
        # Source: https://www.kaggle.com/code/vslaykovsky/pytorch-effnetv2-vertebrae-detection-acc-0-95
        img=dicom.dcmread(path)
        img.PhotometricInterpretation = 'YBR_FULL'
        data = img.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data=(data * 255).astype(np.uint8)
        return cv.cvtColor(data, cv.COLOR_GRAY2RGB), img
    
    def encode_mask(self, mask: np.array):
        """Encode mask. Creates stack of binary masks of shape (N_CLASSES, H, W)

        Args:
            mask (np.array): Non-binary mask of shape (H, W)

        Returns:
            np.array: Binary masks of shape (N_CLASSES, H, W)
        """
        semantic_map = []
        for class_value in self.class_values:
            semantic_map.append(np.equal(mask, class_value).astype(np.uint8))
        return np.stack(semantic_map, axis=-1)
    
    def load_mask(self, uid: str, slice_number: int):
        """Extract and preprocess segmentation mask

        Args:
            uid (str): UID of patient
            slice_number (int): Slice number

        Returns:
            np.array: Binary masks of shape (N_CLASSES, H, W)
        """
        try:
            mask = self.masks[uid][:, :, slice_number]
        except:
            mask = np.zeros((512, 512))
        mask[mask > 7] = 8
        encoded_mask = self.encode_mask(mask)
        return encoded_mask
    
    def __getitem__(self, index: int):
        """Get item by index

        Args:
            index (int): Index value

        Returns:
            Tuple[np.array]: Tuple of image & mask
        """
        record = self.dataframe.iloc[index, :].to_numpy()
        
        img, _ = self.load_dicom(
            os.path.join(self.dicom_path, record[0], (str(record[1]) + '.dcm'))
            )
        mask = self.load_mask(record[0], record[1] - 1)
        
        if self.augmentation:
            sample = self.augmentation(image = img, mask = mask)
            img, mask = sample['image'], sample['mask']
            
        if self.preprocessing:
            sample = self.preprocessing(image = img, mask = mask)
            img, mask = sample['image'], sample['mask']
        
        return img, mask
    
def get_training_augmentation():
    """Returns album.Compose object with set of augmentation instructions

    Returns:
        album.Compose: Augmentation function
    """
    train_transform = [
        album.OneOf(
            [
                album.HorizontalFlip(p=1)
            ],
            p=0.25,
        ),
    ]
    return album.Compose(train_transform)


def to_tensor(x, **kwargs):
    """Image shape & datatype conversion

    Args:
        x (np.array): Original image

    Returns:
        np.array: Reshaped & casted image
    """
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    
    Args:
        preprocessing_fn (callable): data normalization function
        
    Return:
        album.Compose: Preprocessing function
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)