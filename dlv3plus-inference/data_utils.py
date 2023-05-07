import albumentations as album
from tqdm import tqdm
from pathlib import Path
import pydicom as dicom
import os
import zipfile

import config

from typing import IO

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

def check_wrong_resolution():
    """Check DICOM files to find CT scans with resolution other than (512, 512)

    Returns:
        List[str]: List of UIDs with wrong resolution
    """
    wrong_resolution_list = []
    for patient in tqdm(Path(config.DICOM_PATH).iterdir()):
        try:
            md = dicom.dcmread(patient / '10.dcm')
        except:
            wrong_resolution_list.append(patient)
            print(patient)
            continue
        rows, columns = md.Rows, md.Columns
        if rows != 512 or columns != 512:
            wrong_resolution_list.append(patient)
            
    wrong_resolution_list = [str(wrong_uid).split('/')[-1] for wrong_uid in wrong_resolution_list]
    return wrong_resolution_list

def zipdir(path: str, ziph: IO):
    """Zip directory

    Args:
        path (str): Directory path
        ziph (IO): ZipFile object
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))