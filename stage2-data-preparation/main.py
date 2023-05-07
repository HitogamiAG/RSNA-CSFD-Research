import numpy as np
import pandas as pd
import os
import json
from scipy.sparse import load_npz
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

import config
from utils import (
    transform_dataframe,
    load_dicom,
    compute_bounding_rect,
    extract_proportional_elements
)

def main(record):
    _, record = record
    uid = record[0]
    vertebra = record[1]
    state = record[2]
    slices = extract_proportional_elements([int(slc) for slc in record[3].split(',')])
    
    mask = load_npz(os.path.join(config.NEW_SEGMENTATION_PATH, (uid + '.npz')))
    mask = mask[:, [slice_number - 1 for slice_number in slices]].toarray().reshape((512, 512, -1)).astype(np.uint8)
    mask = (mask == vertebra).astype(np.uint8)
    
    bRects = compute_bounding_rect(mask)
    
    imgs = np.zeros((512, 512, len(slices)))
    for i, slc in enumerate(slices):
        imgs[:, :, i] = load_dicom(os.path.join(config.DICOM_PATH, uid, (str(slc) + '.dcm')))
    imgs = np.transpose(imgs, axes=(2, 0, 1))[:, bRects[1]:bRects[3], bRects[0]:bRects[2]].astype(np.uint8)
    masks = np.packbits(np.transpose(mask, axes=(2, 0, 1))[:, bRects[1]:bRects[3], bRects[0]:bRects[2]].astype(bool), axis=None)
    
    np.savez_compressed(os.path.join(config.PREPARED_DATA, f'{uid}_{vertebra}_{state}'), imgs = imgs, masks=masks)

if __name__ == '__main__':
    
    frac_df = pd.read_csv(os.path.join(config.RSNA_PATH, 'train.csv'))
    presented_masks = [mask_name[:-4] for mask_name in os.listdir(str(config.NEW_SEGMENTATION_PATH)) if '.npz' in mask_name]
    frac_df = frac_df[frac_df.StudyInstanceUID.isin(presented_masks)].reset_index(drop=True)
    
    with open(os.path.join(config.NEW_SEGMENTATION_PATH, 'metadata.json'), 'r') as json_file:
        frac_df_metadata = json.load(json_file)
        
    df = transform_dataframe(frac_df, frac_df_metadata)
    df = df[df.Slices.apply(lambda x: len(x.split(',')) > 15)]
    
    if not os.path.isdir(config.PREPARED_DATA):
        os.mkdir(config.PREPARED_DATA)
        
    with ThreadPool(os.cpu_count()) as pool:
        max_ = df.shape[0]
        with tqdm(total=max_) as pbar:
            for _ in pool.imap_unordered(main, df.iterrows()):
                pbar.update()    
    