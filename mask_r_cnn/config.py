# WANDB
PROJECT_NAME = 'test_logger'

# PATHS
# Sources: https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection
#          https://www.kaggle.com/datasets/samuelcortinhas/rsna-2022-spine-fracture-detection-metadata
RSNA_PATH = '../data/rsna-2022-cervical-spine-fracture-detection'
METADATA_PATH = '../data/rsna-2022-spine-fracture-detection-metadata'
DICOM_PATH = '../data/rsna-2022-cervical-spine-fracture-detection/train_images'
SEGMENTATION_PATH = '../data/rsna-2022-cervical-spine-fracture-detection/segmentations'
CHECKPOINT_PATH = './checkpoints'

# Dataset
TRANSFORM = {
    'p_original' : 1,
    'p_hflip' : 0,
    'p_affine' : 0,
    'p_cjitter' : 0
}

# Model
NUM_CLASSES = 8

# Train
SEND_TRAIN_LOGS_EACH_N_BATCHES = 10
