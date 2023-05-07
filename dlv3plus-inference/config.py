# PATHS
# Sources: https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection
#          https://www.kaggle.com/datasets/samuelcortinhas/rsna-2022-spine-fracture-detection-metadata
RSNA_PATH = '../data/rsna-2022-cervical-spine-fracture-detection'
METADATA_PATH = '../data/rsna-2022-spine-fracture-detection-metadata'
DICOM_PATH = '../data/rsna-2022-cervical-spine-fracture-detection/train_images'
SEGMENTATION_PATH = '../data/rsna-2022-cervical-spine-fracture-detection/segmentations'
NEW_SEGMENTATION_PATH = '../data/generated_masks'
CHECKPOINT_PATH = '../dlv3plus/checkpoints'
LOW_HIGH_BOUNDS_CSV_PATH = './ct_lowhigh_bounds.csv'

# Model
ENCODER = 'tu-tf_efficientnetv2_m'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'others']
ACTIVATION = 'softmax2d'

# Inference
DEVICE = 'cuda'
BATCH_SIZE = 16