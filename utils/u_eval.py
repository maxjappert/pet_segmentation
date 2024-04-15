import numpy as np
import torch

def batch_pixel_accuracy(y_true, y_pred):
    """
    Calculate pixel accuracy per image in a batch and return the average accuracy.
    
    Args:
        y_true (np.array): Ground truth masks with shape (batch_size, width, height)
        y_pred (np.array): Predicted masks with shape (batch_size, width, height)
        
    Returns:
        np.array: Pixel accuracy for each image in the batch.
    """
    correct_per_image = np.sum(y_true == y_pred, axis=(1, 2))
    total_per_image = y_true.shape[1] * y_true.shape[2]
    return correct_per_image / total_per_image

def batch_intersection_over_union(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) per image in a batch.
    
    Args:
        y_true (np.array): Ground truth masks with shape (batch_size, width, height)
        y_pred (np.array): Predicted masks with shape (batch_size, width, height)
    
    Returns:
        np.array: IoU for each image in the batch.
    """
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_per_image = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))
    return iou_per_image

def batch_dice_coefficient(y_true, y_pred):
    """
    Calculate the Dice Coefficient per image in a batch.
    
    Args:
        y_true (np.array): Ground truth masks with shape (batch_size, width, height)
        y_pred (np.array): Predicted masks with shape (batch_size, width, height)
    
    Returns:
        np.array: Dice Coefficient for each image in the batch.
    """
    intersection = np.logical_and(y_true, y_pred)
    dice_per_image = 2 * np.sum(intersection, axis=(1, 2)) / (np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2)))
    return dice_per_image

def batch_precision(y_true, y_pred):
    """
    Calculate precision per image in a batch.
    
    Args:
        y_true (np.array): Ground truth masks with shape (batch_size, width, height)
        y_pred (np.array): Predicted masks with shape (batch_size, width, height)
    
    Returns:
        np.array: Precision for each image in the batch.
    """
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1), axis=(1, 2))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0), axis=(1, 2))
    return np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)!=0)

def batch_recall(y_true, y_pred):
    """
    Calculate recall per image in a batch.
    
    Args:
        y_true (np.array): Ground truth masks with shape (batch_size, width, height)
        y_pred (np.array): Predicted masks with shape (batch_size, width, height)
    
    Returns:
        np.array: Recall for each image in the batch.
    """
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1), axis=(1, 2))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1), axis=(1, 2))
    return np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)!=0)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate the model using the provided dataloader and return the average of several metrics.

    Args:
        model (torch.nn.Module): The (pre-loaded) model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device or str): Device to use for computation (e.g., 'cuda' or 'cpu').

    Returns:
        dict: Dictionary containing the mean accuracy, IoU, Dice coefficient, precision, and recall.
    """
    model.eval()

    total_accuracy = []
    total_iou = []
    total_dice = []
    total_precision = []
    total_recall = []

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        labels = labels.cpu().numpy()
        
        accuracy = batch_pixel_accuracy(labels, preds)
        iou = batch_intersection_over_union(labels, preds)
        dice = batch_dice_coefficient(labels, preds)
        precision = batch_precision(labels, preds)
        recall = batch_recall(labels, preds)

        total_accuracy.append(accuracy)
        total_iou.append(iou)
        total_dice.append(dice)
        total_precision.append(precision)
        total_recall.append(recall)
                
    mean_accuracy = np.mean(np.concatenate(total_accuracy))
    mean_iou = np.mean(np.concatenate(total_iou))
    mean_dice = np.mean(np.concatenate(total_dice))
    mean_precision = np.mean(np.concatenate(total_precision))
    mean_recall = np.mean(np.concatenate(total_recall))

    return {
        'accuracy': mean_accuracy,
        'iou': mean_iou,
        'dice_coefficient': mean_dice,
        'precision': mean_precision,
        'recall': mean_recall
    }