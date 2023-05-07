import torch
import pandas as pd
import os
import json
import numpy as np
from tqdm import tqdm
import zipfile

from model import initialize_model, load_weights
from data_utils import (
    check_wrong_resolution,
    get_preprocessing,
    zipdir)       
from utils import (
    set_seeds,
    save_matrix, 
    get_slice_n_from_path,
    load_dicom,
    clear_mask)
import config

from typing import List, Callable

def inference(
    model: torch.nn.Module,
    preprocessing_fn: Callable,
    wrong_resolution_list: List[str],
    bounds_df: pd.DataFrame,
    segm_uids: List[str]
):
    """Generate masks for the whole dataset

    Args:
        wrong_resolution_list (List[str]): UIDs with resolution other than (512, 512)
        bounds_df (pd.DataFrame): Limits vertebra occurencies by slice numbers.
                                  Detailts in ct-low-high-bounds-kaggle-code.ipynb
        segm_uids (List[str]): UIDs with hand-labeled segmentations
    """
    
    if not os.path.isdir(config.NEW_SEGMENTATION_PATH):
        os.mkdir(config.NEW_SEGMENTATION_PATH)
        
    json_dict_name = os.path.join(config.NEW_SEGMENTATION_PATH, 'metadata.json')
    json_dict = {}
    
    preprocess = get_preprocessing(preprocessing_fn)
    
    for _, patient_record in tqdm(bounds_df.iterrows()):
        patient = patient_record[0]
        
        if patient in wrong_resolution_list or patient in segm_uids:
            continue
            
        patient_path = os.path.join(config.DICOM_PATH, patient)
        
        list_slices = sorted([get_slice_n_from_path(file) for file in os.listdir(patient_path)])
        num_slices = len(list_slices)
        
        mask = np.zeros((512, 512, num_slices))
        batch_counter = 0
        
        while len(list_slices):
            slices = list_slices[:config.BATCH_SIZE]
            list_slices = list_slices[config.BATCH_SIZE:]
            imgs = np.zeros((len(slices), 3, 512, 512))
            for ind, slice_ in enumerate(slices):
                img, _ = load_dicom(patient_path / (str(slice_) + '.dcm'))
                img = preprocess(image = img)['image']
                imgs[ind, :, :, :] = img
            
            imgs = torch.FloatTensor(imgs).to('cuda')
            
            with torch.cuda.amp.autocast():
                predictions = model(imgs)
            predictions = np.argmax(predictions.cpu(), axis=1)
            mask[:, :, (batch_counter * config.BATCH_SIZE) : ((batch_counter * config.BATCH_SIZE) + len(slices))] = \
                np.transpose(predictions, axes = [1, 2, 0])
            batch_counter += 1
            
        mask = mask.astype(np.uint8)
        for i in range(mask.shape[2]):
            mask[:, :, i] = clear_mask(mask[:, :, i], patient_record, i + 1)
            
        save_matrix(mask, f'/kaggle/working/dataset/{patient}.npz')
        json_dict[patient] = {}
        
        for i in range(mask.shape[2]):
            vertebrae = np.unique(mask[:, :, i])[1:].tolist()
            for vertebra in vertebrae:
                if vertebra in json_dict[patient]:
                    json_dict[patient][vertebra].append(i)
                else:
                    json_dict[patient][vertebra] = []
                    json_dict[patient][vertebra].append(i)
                    
    with open(json_dict_name, 'w') as json_file:
        json.dump(json_dict, json_file)
        
    with zipfile.ZipFile('/kaggle/working/dataset.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('/kaggle/working/dataset', zipf)
    

if __name__ == '__main__':
    
    set_seeds()
    model, preprocessing_fn = initialize_model()
    model = load_weights(model)
    
    wrong_resolution_list = check_wrong_resolution()
    bounds_df = pd.read_csv(config.LOW_HIGH_BOUNDS_CSV_PATH, index_col=0)
    segm_uids = [uid.replace('.nii', '') for uid in os.listdir(config.SEGMENTATION_PATH)]
    
    inference(model,
              preprocessing_fn,
              wrong_resolution_list,
              bounds_df,
              segm_uids)