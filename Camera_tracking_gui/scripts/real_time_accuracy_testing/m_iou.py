import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from densenet import DenseNet2D
from mobilenet_v1 import MobileNet2D_V1
from PIL import Image
from torchvision import transforms
import utils
import gt_masks
from glob import glob


def init_model(model_path, device, dense_net=False):
    
    try:
        print(f"Using device: {device}")
        # Create model instance
        if dense_net:
            model = DenseNet2D(dropout=True, prob=0.2)
        else: 
            model = MobileNet2D_V1(dropout=True,prob=0.2)
        model = model.to(device)
        
        # Load the state dictionary
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
            
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return

def process_image(image, model, device):
    # prepare frame for model
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray_frame)
    img_tensor = transform(pil_image)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
    
    # Get prediction map using utils.get_predictions
    pred_map = utils.get_predictions(output)
    pred_img = pred_map.cpu().numpy().squeeze(0)

    
    pred_masks = {
        'pupil': pred_img == 3,
        'iris': pred_img == 2,
        'sclera': pred_img == 1
    }

    return pred_masks

def calculate_iou(pred_mask, gt_mask):
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_mean_iou(pred_masks, gt_masks):
    
    iou_scores = {}
    for key in pred_masks.keys():
        iou_scores[key] = calculate_iou(pred_masks[key], gt_masks[key])
    return iou_scores

def main():
    # Load model
    model_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILE_V1.pkl"
    dense_net_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\dense_net9.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(model_path, device, False)
    dense_net_model = init_model(dense_net_path, device, True)
    # load pictures
    ds_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\real_time_ds"
    imageFiles = glob(os.path.join(ds_path, '*.jpg'))
    jsonFiles = glob(os.path.join(ds_path, '*.json'))
    imageFiles.sort()
    jsonFiles.sort()
    dataset = []

    ## Create dataset of image and json files
    for i in range(len(imageFiles)):
        image_path = imageFiles[i]
        json_path = jsonFiles[i]
        dataset.append((image_path, json_path))

    all_iou_scores = np.zeros((len(dataset), 3))
    all_iou_scores_dense = np.zeros((len(dataset), 3))
    i = 0
    for image_path, json_path in dataset:
        # Load image
        image = cv2.imread(image_path)
        pupil_mask_gt, iris_mask_gt, sclera_mask_gt = gt_masks.gt_masks(json_path)
        ## Convert masks to binary
        pupil_mask_gt = pupil_mask_gt>0
        iris_mask_gt = iris_mask_gt>0
        sclera_mask_gt = sclera_mask_gt>0

        pred_masks = process_image(image, model, device)
        pred_masks_dense = process_image(image, dense_net_model, device)

        # Our network
        pupil_iou = calculate_iou(pred_masks['pupil'], pupil_mask_gt)
        iris_iou = calculate_iou(pred_masks['iris'], iris_mask_gt)  
        sclera_iou = calculate_iou(pred_masks['sclera'], sclera_mask_gt)
        all_iou_scores[i, 0] = pupil_iou
        all_iou_scores[i, 1] = iris_iou
        all_iou_scores[i, 2] = sclera_iou
        print(f"Image: {os.path.basename(image_path)}, Pupil IoU: {pupil_iou:.4f}, Iris IoU: {iris_iou:.4f}, Sclera IoU: {sclera_iou:.4f}")

        # DenseNet
        pupil_iou_dense = calculate_iou(pred_masks_dense['pupil'], pupil_mask_gt)
        iris_iou_dense = calculate_iou(pred_masks_dense['iris'], iris_mask_gt)
        sclera_iou_dense = calculate_iou(pred_masks_dense['sclera'], sclera_mask_gt)
        all_iou_scores_dense[i, 0] = pupil_iou_dense
        all_iou_scores_dense[i, 1] = iris_iou_dense
        all_iou_scores_dense[i, 2] = sclera_iou_dense
        print(f"DenseNet - Image: {os.path.basename(image_path)}, Pupil IoU: {pupil_iou_dense:.4f}, Iris IoU: {iris_iou_dense:.4f}, Sclera IoU: {sclera_iou_dense:.4f}")
        i += 1
    print(f"Mean IoU: Pupil: {np.mean(all_iou_scores[:, 0]):.4f}, Iris: {np.mean(all_iou_scores[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores[:, 2]):.4f}")
    print(f"Mean DenseNet IoU: Pupil: {np.mean(all_iou_scores_dense[:, 0]):.4f}, Iris: {np.mean(all_iou_scores_dense[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores_dense[:, 2]):.4f}")

if __name__ == "__main__":
    main()
