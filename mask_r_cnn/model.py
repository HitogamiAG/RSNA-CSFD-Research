import torch
from torch import nn

import torchvision as tv
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.resnet import resnet101
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNHeads
from torchvision.models.detection.faster_rcnn import _default_anchorgen, RPNHead, FastRCNNConvFCHead

import os
import wget

import config

def get_backbone():
    """Initializes ResNet101 backbone with COCO pre-trained weights

    Returns:
        nn.Module: Backbone
    """
    dlv3 = tv.models.segmentation.deeplabv3.deeplabv3_resnet101(weights = tv.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)

    rn101_original = tv.models.resnet.resnet101()
    del rn101_original.fc

    rn101_original.load_state_dict(dlv3.get_submodule('backbone').state_dict())
    return rn101_original

def download_maskrcnn_weights():
    """Download Mask R-CNN COCO pre-trained weights
    """
    if not os.path.isfile('maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'):
        url = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'
        wget.download(url)
        
def load_weights(model: nn.Module) -> nn.Module:
    """Load pre-trained weights to Mask R-CNN

    Args:
        model (nn.Module): Mask R-CNN model

    Returns:
        nn.Module: Mask R-CNN with pre-trained weights
    """
    download_maskrcnn_weights()
    weights = torch.load('maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth')
    model_keys = set(model.state_dict().keys())
    weight_keys = set(weights.keys())

    model_shapes = {key : value.shape for key, value in model.state_dict().items()}
    weight_shapes = {key : value.shape for key, value in weights.items()}

    mapping = []

    for key in model_keys:
        if not (key.split('.')[0] == 'backbone' and key.split('.')[1] == 'body') and key in weight_keys and weight_shapes[key] == model_shapes[key]:
            mapping.append(key)
            
    model_dict = model.state_dict()
    for key in mapping:
        model_dict[key] = weights[key]

    model.load_state_dict(model_dict)
    return model

def initialize_model(num_classes: int = config.NUM_CLASSES) -> torch.nn.Module:
    """Initializes model with configured parameters

    Args:
        num_classes (int, optional): Number of classes. Defaults to config.NUM_CLASSES.

    Returns:
        torch.nn.Module: Mask R-CNN model
    """
    backbone = get_backbone()
    
    trainable_backbone_layers = None
    is_trained = False

    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    # Backbone
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer = nn.BatchNorm2d)

    # Anchor generator
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # RPN Module
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)

    # Box Module
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )

    # Mask Module
    mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
    mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d)
    
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        
        # transform parameters ------------------
        min_size=500,
        max_size=600,
        image_mean=None,
        image_std=None,
        
        # RPN parameters -------------------------
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        rpn_pre_nms_top_n_train=500 * 4,
        rpn_pre_nms_top_n_test=250 * 4,
        rpn_post_nms_top_n_train=500 * 4,
        rpn_post_nms_top_n_test=250 * 4,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        
        # Box parameters ---------------
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        
        # must be None when num_classes specified
        box_predictor=None,
        
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        
        # Mask parameters --------------------------
        mask_roi_pool=mask_roi_pool,
        mask_head=mask_head,
        
        # must be None when num_classes specified
        mask_predictor=None
    )
    
    model = load_weights(model)
    
    return model