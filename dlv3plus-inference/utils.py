import torch
import numpy as np
import pandas as pd

import pydicom as dicom
import cv2 as cv

from scipy.sparse import csr_matrix, save_npz

def set_seeds():
    """Set seeds to random generators
    """
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def clear_mask(mask: np.array, patient_record: pd.Series, slice_number: int):
    """Clear generated mask from tiny objects & apply bounds for vertebra

    Args:
        mask (np.array): Generated mask
        patient_record (pd.Series): Record from dataframe
        slice_number (int): Number of slice

    Returns:
        np.array: Cleared mask
    """
    labels, counts = np.unique(mask, return_counts=True)
    labels, counts = labels[1:], counts[1:]
    
    match_dict = {}
    
    for label, count in zip(labels, counts):
        if count < 50:
            mask[mask == label] = 0
            continue
        
        if label > 0 and label < 8:
            lb = patient_record[f'C{label}_lb']
            hb = patient_record[f'C{label}_hb']
        else:
            continue
        
        new_label = label
        
        while lb > slice_number or hb < slice_number:
            if lb < slice_number:
                new_label -= 1
            elif hb < slice_number:
                new_label += 1
            else:
                break
            
            if new_label > 0 and new_label < 8:
                lb = patient_record[f'C{new_label}_lb']
                hb = patient_record[f'C{new_label}_hb']
            else:
                break
        
        if new_label != label:
            mask[mask == label] == new_label
    
    return mask

def get_slice_n_from_path(path: str):
    """Extract from DICOM file path the number of slice

    Args:
        path (str): Path to .dcm file

    Returns:
        int: Slice number
    """
    return int(str(path).split('/')[-1].split('.')[-2])

def load_dicom(path: str):
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

def save_matrix(mask: np.array, save_path: str):
    """Save sparse matrix as .npz file

    Args:
        mask (np.array): Generated mask
        save_path (str): Path to file
    """
    sparse_matrix = csr_matrix(mask.reshape(-1, mask.shape[-1]))
    save_npz(save_path, sparse_matrix)