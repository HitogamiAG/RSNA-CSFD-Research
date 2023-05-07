import pandas as pd
import numpy as np
import cv2 as cv
import pydicom as dicom

from typing import Dict, Tuple, List

def transform_dataframe(frac_df: pd.DataFrame,
                        frac_df_metadata: Dict[str, Dict[str, List[int]]]) -> pd.DataFrame:
    """Transforms dataframe so than each record represents
       one vertebra instead of one patient UID

    Args:
        frac_df (pd.DataFrame): Fracture dataframe
        frac_df_metadata (pd.DataFrame): Metadata from generated masks

    Returns:
        pd.DataFrame: Transformed dataframe
    """
    df = {'StudyInstanceUID' : [],
      'Vertebra' : [],
      'State' : [],
      'Slices' : []}

    for _, record in frac_df.iterrows():
        for ind, (_, state) in enumerate(record[2:].items()):
            try:
                df['Slices'].append(','.join([str(slc) for slc in frac_df_metadata[record[0]][str(ind+1)]]))
                df['StudyInstanceUID'].append(record[0])
                df['Vertebra'].append(ind+1)
                df['State'].append(state)
            except:
                continue
        
    return pd.DataFrame(df)

def extract_proportional_elements(arr: List[int], T_size: int = 32):
    """From list of slices extract N slices with uniform dictribution

    Args:
        arr (List[int]): List of slice numbers
        T_size (int): Number of slices to extract. Defaults to 32.

    Returns:
        List[int]: Extracted slice numbers
    """
    length_of_array = len(arr)

    if length_of_array == T_size:
        return arr
    elif length_of_array > T_size:
        step_size = length_of_array / T_size
        current_index = 0
        result_list = []
        for i in range(T_size):
            next_index = int(current_index + step_size)
            if next_index >= length_of_array:
                next_index = length_of_array - 1
            result_list.append(arr[next_index])
            current_index = next_index
        return result_list
    else:
        result_list = []
        num_copies = T_size // length_of_array
        for i in range(num_copies):
            result_list.extend(arr)

        remainder = T_size - len(result_list)

        if remainder != 0:
            step_size = length_of_array / remainder
            current_index = 0
            for i in range(remainder):
                next_index = current_index + step_size
                if next_index >= length_of_array:
                    next_index = length_of_array - 1
                result_list.append(arr[int(next_index)])
                current_index = next_index
        return sorted(result_list)

def compute_bounding_rect(mask: np.array) -> Tuple[int]:
    """Computes bounding box with size (224, 224) for segmentation mask

    Args:
        mask (np.array): Segmentation mask

    Returns:
        Tuple[int]: Bounding box coordinates
    """
    # compute rects
    bRects = [cv.boundingRect(mask[:, :, i]) for i in range(5, 32, 5)]
    # compute centers
    bRects = [(rect[0] + int(rect[2] / 2), rect[1] + int(rect[3] / 2)) for rect in bRects]
    # compute mean
    x, y = np.array(bRects).mean(axis=0).astype(int)
    # calculate compensation
    x_compensation = max(-(x - 112), 0) + min(-(x + 112 - 512), 0)
    y_compensation = max(-(y - 112), 0) + min(-(y + 112 - 512), 0)
    # apply compensation and use 
    x, y = x + x_compensation, y + y_compensation
    x1, y1, x2, y2 = (
        x - 112, y - 112, x + 112, y + 112
    )
    return (x1, y1, x2, y2)

def load_dicom(path: str):
    """Load DICOM (.dcm) file

        Args:
            path (str): Path to DICOM file

        Returns:
            Tuple[np.array, Dict[str, Any]]: Returns image & metadata
    """
    # Source: https://www.kaggle.com/code/vslaykovsky/pytorch-effnetv2-vertebrae-detection-acc-0-95
    try:
        img=dicom.dcmread(path)
    except:
        print('Error at load_dicom')
        return np.zeros((512, 512))
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data=(data * 255).astype(np.uint8)
    return data