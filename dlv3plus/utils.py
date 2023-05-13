import torch
import numpy as np

def set_seeds():
    """Set seeds to random generators
    """
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_pixel_accuracy(predictions, ground_truth):
    """
    Get pixel accuracy for each class represented in a NumPy array and total pixel accuracy for the whole mask.

    Args:
    predictions: NumPy array of shape (512, 512). NumPy array represent semantic segmentation model predictions. The values of array are the classes which the pixels belong to. 0 is background.
    ground_truth: NumPy array of shape (512, 512). NumPy array represent ground truth segmentation mask. The values of array are the classes which the pixels belong to. 0 is background.

    Returns:
    Dictionary of pixel accuracies for each class.
    """

    # Check the shapes of the input arrays.
    assert predictions.shape == ground_truth.shape

    # Get the unique classes in the predictions and ground truth masks.
    classes = np.concatenate([np.unique(predictions), np.unique(ground_truth)], axis=0)
    classes = classes[classes != 0]

    # Initialize the dictionary of pixel accuracies.
    pixel_accuracies = {}

    # Loop over the classes.
    for class_id in classes:
        # Get the predictions and ground truth masks for the current class.
        prediction_mask = predictions == class_id
        ground_truth_mask = ground_truth == class_id

        # Get the number of pixels that are correctly classified.
        correct_pixels = np.sum(prediction_mask & ground_truth_mask)

        # Get the total number of pixels in the current class.
        total_pixels = np.max([np.sum(prediction_mask), np.sum(ground_truth_mask)])

        # Calculate the pixel accuracy for the current class.
        
        #pixel_accuracy = correct_pixels / total_pixels

        # Add the pixel accuracy for the current class to the dictionary.
        pixel_accuracies[class_id] = [correct_pixels, total_pixels]

    # Get the total pixel accuracy for the whole mask.
    pixel_accuracies['total'] = [np.sum(predictions == ground_truth), predictions.size]

    # Return the dictionary of pixel accuracies for each class and the total pixel accuracy for the whole mask.
    return pixel_accuracies

def get_iou(predictions, ground_truth):
    """
    Get Intersection over Union (IoU) for each class represented in a NumPy array and total IoU for the whole mask.

    Args:
    predictions: NumPy array of shape (512, 512). NumPy array represent semantic segmentation model predictions. The values of array are the classes which the pixels belong to. 0 is background.
    ground_truth: NumPy array of shape (512, 512). NumPy array represent ground truth segmentation mask. The values of array are the classes which the pixels belong to. 0 is background.

    Returns:
    Dictionary of IoU values for each class.
    """

    # Check the shapes of the input arrays.
    assert predictions.shape == ground_truth.shape

    # Get the unique classes in the predictions and ground truth masks.
    classes = np.concatenate([np.unique(predictions), np.unique(ground_truth)], axis=0)
    classes = classes[classes != 0]

    # Initialize the dictionary of IoU values.
    iou_values = {}

    # Loop over the classes.
    for class_id in classes:
        # Get the predictions and ground truth masks for the current class.
        prediction_mask = predictions == class_id
        ground_truth_mask = ground_truth == class_id

        # Get the intersection of the predictions and ground truth masks.
        intersection = np.sum(prediction_mask & ground_truth_mask)

        # Get the union of the predictions and ground truth masks.
        union = np.sum(prediction_mask) + np.sum(ground_truth_mask) - intersection

        # Calculate the IoU for the current class.
        #iou = intersection / union

        # Add the IoU for the current class to the dictionary.
        iou_values[class_id] = [intersection, union]

    # Return the dictionary of IoU values for each class and the total IoU for the whole mask.
    return iou_values

def get_dc(predictions, ground_truth):
    """
    Get Dice Coefficient (DC) for each class represented in a NumPy array and total DC for the whole mask.

    Args:
    predictions: NumPy array of shape (512, 512). NumPy array represent semantic segmentation model predictions. The values of array are the classes which the pixels belong to. 0 is background.
    ground_truth: NumPy array of shape (512, 512). NumPy array represent ground truth segmentation mask. The values of array are the classes which the pixels belong to. 0 is background.

    Returns:
    Dictionary of DC values for each class.
    """

    # Check the shapes of the input arrays.
    assert predictions.shape == ground_truth.shape

    # Get the unique classes in the predictions and ground truth masks.
    classes = np.concatenate([np.unique(predictions), np.unique(ground_truth)], axis=0)
    classes = classes[classes != 0]

    # Initialize the dictionary of DC values.
    dc_values = {}

    # Loop over the classes.
    for class_id in classes:
        # Get the predictions and ground truth masks for the current class.
        prediction_mask = predictions == class_id
        ground_truth_mask = ground_truth == class_id

        # Get the intersection of the predictions and ground truth masks.
        intersection = np.sum(prediction_mask & ground_truth_mask)

        # Get the union of the predictions and ground truth masks.
        union = np.sum(prediction_mask) + np.sum(ground_truth_mask)

        # Calculate the DC for the current class.
        #dc = (2 * intersection) / (union + 1e-12)

        # Add the DC for the current class to the dictionary.
        dc_values[class_id] = [(2 * intersection), (union + 1e-12)]
    
    # Return the dictionary of DC values for each class and the total DC for the whole mask.
    return dc_values