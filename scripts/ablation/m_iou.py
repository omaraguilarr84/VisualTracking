import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from densenet import DenseNet2D
from mobilenet_v1 import MobileNet2D_V1
from mobilenet_v2 import MobileNet2D_V2
from mobilenet_v3 import MobileNet2D_V3
from mobilenet_v4 import MobileNet2D_V4
from mobilenet_v5 import MobileNet2D_V5
from mobilenet_v1_AP import MobileNet2D_V1_AP
from mobilenet_v2_AP import MobileNet2D_V2_AP
from mobilenet_v3_AP import MobileNet2D_V3_AP
from mobilenet_v4_AP import MobileNet2D_V4_AP
from mobilenet_v5_AP import MobileNet2D_V5_AP
from densenet_og_AP import DenseNet2D_AP
from PIL import Image
from torchvision import transforms
import utils
import gt_masks
from glob import glob
from matplotlib import rcParams



def init_model(model_path, device, modeltype=0):
    
    try:
        print(f"Using device: {device}")
        # Create model instance
        if modeltype == 0:
            model = MobileNet2D_V1(dropout=True, prob=0.2)
        elif modeltype == 1:
            model = MobileNet2D_V2(dropout=True, prob=0.2)
        elif modeltype == 2:
            model = MobileNet2D_V3(dropout=True, prob=0.2)
        elif modeltype == 3:
            model = MobileNet2D_V4(dropout=True, prob=0.2)
        elif modeltype == 4:
            model = MobileNet2D_V5(dropout=True, prob=0.2)
        elif modeltype == 5:
            model = DenseNet2D(dropout=True, prob=0.2)
        elif modeltype == 6:
            model = MobileNet2D_V1_AP(dropout=True, prob=0.2)
        elif modeltype == 7:
            model = MobileNet2D_V2_AP(dropout=True, prob=0.2)
        elif modeltype == 8:
            model = MobileNet2D_V3_AP(dropout=True, prob=0.2)
        elif modeltype == 9:
            model = MobileNet2D_V4_AP(dropout=True, prob=0.2)
        elif modeltype == 10:
            model = MobileNet2D_V5_AP(dropout=True, prob=0.2)
        elif modeltype == 11:
            model = DenseNet2D_AP(dropout=True, prob=0.2)
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
    model2_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILE_V2.pkl"
    model3_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILE_V3.pkl"
    model4_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILE_V4.pkl"
    model5_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILE_V5.pkl"
    model6_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILE_V1_AP_ROI9.pkl"
    model7_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILENET_V2_AP_ROI9.pkl"
    model8_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILENET_V3_AP_ROI9.pkl"
    model9_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILENET_V4_AP_ROI9.pkl"
    model10_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_MOBILENET_V5_AP_ROI9.pkl"
    dense_net_ap_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\TRAIN_OG_AP_ROI9.pkl"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(model_path, device, 0)
    model2 = init_model(model2_path, device, 1)
    model3 = init_model(model3_path, device, 2)
    model4 = init_model(model4_path, device, 3)
    model5 = init_model(model5_path, device, 4)
    dense_net_model = init_model(dense_net_path, device, 5)
    model6 = init_model(model6_path, device, 6)
    model7 = init_model(model7_path, device, 7)
    model8 = init_model(model8_path, device, 8)
    model9 = init_model(model9_path, device, 9)
    model10 = init_model(model10_path, device, 10)
    dense_net_ap_model = init_model(dense_net_ap_path, device, 11)
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
    all_iou_scores2 = np.zeros((len(dataset), 3))
    all_iou_scores3 = np.zeros((len(dataset), 3))
    all_iou_scores4 = np.zeros((len(dataset), 3))
    all_iou_scores5 = np.zeros((len(dataset), 3))
    all_iou_scores6 = np.zeros((len(dataset), 3))
    all_iou_scores7 = np.zeros((len(dataset), 3))
    all_iou_scores8 = np.zeros((len(dataset), 3))
    all_iou_scores9 = np.zeros((len(dataset), 3))
    all_iou_scores10 = np.zeros((len(dataset), 3))
    all_iou_scores_dense_ap = np.zeros((len(dataset), 3))
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
        pred_masks2 = process_image(image, model2, device)
        pred_masks3 = process_image(image, model3, device)
        pred_masks4 = process_image(image, model4, device)
        pred_masks5 = process_image(image, model5, device)
        pred_masks6 = process_image(image, model6, device)
        pred_masks7 = process_image(image, model7, device)
        pred_masks8 = process_image(image, model8, device)
        pred_masks9 = process_image(image, model9, device)
        pred_masks10 = process_image(image, model10, device)
        pred_masks_dense_ap = process_image(image, dense_net_ap_model, device)

        # MobileNetV1
        pupil_iou = calculate_iou(pred_masks['pupil'], pupil_mask_gt)
        iris_iou = calculate_iou(pred_masks['iris'], iris_mask_gt)  
        sclera_iou = calculate_iou(pred_masks['sclera'], sclera_mask_gt)
        all_iou_scores[i, 0] = pupil_iou
        all_iou_scores[i, 1] = iris_iou
        all_iou_scores[i, 2] = sclera_iou
        print(f"Image: {os.path.basename(image_path)}, Pupil IoU: {pupil_iou:.4f}, Iris IoU: {iris_iou:.4f}, Sclera IoU: {sclera_iou:.4f}")
        # MobileNetV2
        pupil_iou2 = calculate_iou(pred_masks2['pupil'], pupil_mask_gt)
        iris_iou2 = calculate_iou(pred_masks2['iris'], iris_mask_gt)
        sclera_iou2 = calculate_iou(pred_masks2['sclera'], sclera_mask_gt)
        all_iou_scores2[i, 0] = pupil_iou2
        all_iou_scores2[i, 1] = iris_iou2
        all_iou_scores2[i, 2] = sclera_iou2
        # MobileNetV3
        pupil_iou3 = calculate_iou(pred_masks3['pupil'], pupil_mask_gt)
        iris_iou3 = calculate_iou(pred_masks3['iris'], iris_mask_gt)
        sclera_iou3 = calculate_iou(pred_masks3['sclera'], sclera_mask_gt)
        all_iou_scores3[i, 0] = pupil_iou3
        all_iou_scores3[i, 1] = iris_iou3
        all_iou_scores3[i, 2] = sclera_iou3
        # MobileNetV4
        pupil_iou4 = calculate_iou(pred_masks4['pupil'], pupil_mask_gt)
        iris_iou4 = calculate_iou(pred_masks4['iris'], iris_mask_gt)
        sclera_iou4 = calculate_iou(pred_masks4['sclera'], sclera_mask_gt)
        all_iou_scores4[i, 0] = pupil_iou4
        all_iou_scores4[i, 1] = iris_iou4
        all_iou_scores4[i, 2] = sclera_iou4
        # MobileNetV5
        pupil_iou5 = calculate_iou(pred_masks5['pupil'], pupil_mask_gt)
        iris_iou5 = calculate_iou(pred_masks5['iris'], iris_mask_gt)
        sclera_iou5 = calculate_iou(pred_masks5['sclera'], sclera_mask_gt)
        all_iou_scores5[i, 0] = pupil_iou5
        all_iou_scores5[i, 1] = iris_iou5
        all_iou_scores5[i, 2] = sclera_iou5
        # DenseNet
        pupil_iou_dense = calculate_iou(pred_masks_dense['pupil'], pupil_mask_gt)
        iris_iou_dense = calculate_iou(pred_masks_dense['iris'], iris_mask_gt)
        sclera_iou_dense = calculate_iou(pred_masks_dense['sclera'], sclera_mask_gt)
        all_iou_scores_dense[i, 0] = pupil_iou_dense
        all_iou_scores_dense[i, 1] = iris_iou_dense
        all_iou_scores_dense[i, 2] = sclera_iou_dense
        #MobileNetV1_AP
        pupil_iou6 = calculate_iou(pred_masks6['pupil'], pupil_mask_gt)
        iris_iou6 = calculate_iou(pred_masks6['iris'], iris_mask_gt)
        sclera_iou6 = calculate_iou(pred_masks6['sclera'], sclera_mask_gt)
        all_iou_scores6[i, 0] = pupil_iou6
        all_iou_scores6[i, 1] = iris_iou6
        all_iou_scores6[i, 2] = sclera_iou6
        #MobileNetV2_AP
        pupil_iou7 = calculate_iou(pred_masks7['pupil'], pupil_mask_gt)
        iris_iou7 = calculate_iou(pred_masks7['iris'], iris_mask_gt)
        sclera_iou7 = calculate_iou(pred_masks7['sclera'], sclera_mask_gt)
        all_iou_scores7[i, 0] = pupil_iou7
        all_iou_scores7[i, 1] = iris_iou7
        all_iou_scores7[i, 2] = sclera_iou7
        #MobileNetV3_AP
        pupil_iou8 = calculate_iou(pred_masks8['pupil'], pupil_mask_gt)
        iris_iou8 = calculate_iou(pred_masks8['iris'], iris_mask_gt)
        sclera_iou8 = calculate_iou(pred_masks8['sclera'], sclera_mask_gt)
        all_iou_scores8[i, 0] = pupil_iou8
        all_iou_scores8[i, 1] = iris_iou8
        all_iou_scores8[i, 2] = sclera_iou8
        #MobileNetV4_AP
        pupil_iou9 = calculate_iou(pred_masks9['pupil'], pupil_mask_gt)
        iris_iou9 = calculate_iou(pred_masks9['iris'], iris_mask_gt)
        sclera_iou9 = calculate_iou(pred_masks9['sclera'], sclera_mask_gt)
        all_iou_scores9[i, 0] = pupil_iou9
        all_iou_scores9[i, 1] = iris_iou9
        all_iou_scores9[i, 2] = sclera_iou9
        #MobileNetV5_AP
        pupil_iou10 = calculate_iou(pred_masks10['pupil'], pupil_mask_gt)
        iris_iou10 = calculate_iou(pred_masks10['iris'], iris_mask_gt)
        sclera_iou10 = calculate_iou(pred_masks10['sclera'], sclera_mask_gt)
        all_iou_scores10[i, 0] = pupil_iou10
        all_iou_scores10[i, 1] = iris_iou10
        all_iou_scores10[i, 2] = sclera_iou10
        #DenseNet_AP
        pupil_iou_dense_ap = calculate_iou(pred_masks_dense_ap['pupil'], pupil_mask_gt)
        iris_iou_dense_ap = calculate_iou(pred_masks_dense_ap['iris'], iris_mask_gt)
        sclera_iou_dense_ap = calculate_iou(pred_masks_dense_ap['sclera'], sclera_mask_gt)
        all_iou_scores_dense_ap[i, 0] = pupil_iou_dense_ap
        all_iou_scores_dense_ap[i, 1] = iris_iou_dense_ap
        all_iou_scores_dense_ap[i, 2] = sclera_iou_dense_ap
        
        print(f"DenseNet - Image: {os.path.basename(image_path)}, Pupil IoU: {pupil_iou_dense:.4f}, Iris IoU: {iris_iou_dense:.4f}, Sclera IoU: {sclera_iou_dense:.4f}")
        i += 1
    print(f"Mean IoU: Pupil: {np.mean(all_iou_scores[:, 0]):.4f}, Iris: {np.mean(all_iou_scores[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores[:, 2]):.4f}")
    print(f"Mean DenseNet IoU: Pupil: {np.mean(all_iou_scores_dense[:, 0]):.4f}, Iris: {np.mean(all_iou_scores_dense[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores_dense[:, 2]):.4f}")
    print(f"Mean MobileNetV2 IoU: Pupil: {np.mean(all_iou_scores2[:, 0]):.4f}, Iris: {np.mean(all_iou_scores2[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores2[:, 2]):.4f}")
    print(f"Mean MobileNetV3 IoU: Pupil: {np.mean(all_iou_scores3[:, 0]):.4f}, Iris: {np.mean(all_iou_scores3[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores3[:, 2]):.4f}")
    print(f"Mean MobileNetV4 IoU: Pupil: {np.mean(all_iou_scores4[:, 0]):.4f}, Iris: {np.mean(all_iou_scores4[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores4[:, 2]):.4f}")
    print(f"Mean MobileNetV5 IoU: Pupil: {np.mean(all_iou_scores5[:, 0]):.4f}, Iris: {np.mean(all_iou_scores5[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores5[:, 2]):.4f}")
    print(f"Mean MobileNetV1_AP IoU: Pupil: {np.mean(all_iou_scores6[:, 0]):.4f}, Iris: {np.mean(all_iou_scores6[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores6[:, 2]):.4f}")
    print(f"Mean MobileNetV2_AP IoU: Pupil: {np.mean(all_iou_scores7[:, 0]):.4f}, Iris: {np.mean(all_iou_scores7[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores7[:, 2]):.4f}")
    print(f"Mean MobileNetV3_AP IoU: Pupil: {np.mean(all_iou_scores8[:, 0]):.4f}, Iris: {np.mean(all_iou_scores8[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores8[:, 2]):.4f}")
    print(f"Mean MobileNetV4_AP IoU: Pupil: {np.mean(all_iou_scores9[:, 0]):.4f}, Iris: {np.mean(all_iou_scores9[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores9[:, 2]):.4f}")
    print(f"Mean MobileNetV5_AP IoU: Pupil: {np.mean(all_iou_scores10[:, 0]):.4f}, Iris: {np.mean(all_iou_scores10[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores10[:, 2]):.4f}")
    print(f"Mean DenseNet_AP IoU: Pupil: {np.mean(all_iou_scores_dense_ap[:, 0]):.4f}, Iris: {np.mean(all_iou_scores_dense_ap[:, 1]):.4f}, Sclera: {np.mean(all_iou_scores_dense_ap[:, 2]):.4f}")
    
    # Set Times New Roman as the font
    rcParams['font.family'] = 'Times New Roman'

    # Example IoU scores for each model
    models = ['SERTnetV1', 'SERTnetV2', 'SERTnetV3', 'SERTnetV4', 'SERTnetV5', 'RITnet']
    pupil_ious = [np.mean(all_iou_scores[:, 0]), np.mean(all_iou_scores2[:, 0]), np.mean(all_iou_scores3[:, 0]),
                np.mean(all_iou_scores4[:, 0]), np.mean(all_iou_scores5[:, 0]), np.mean(all_iou_scores_dense[:, 0])]
    iris_ious = [np.mean(all_iou_scores[:, 1]), np.mean(all_iou_scores2[:, 1]), np.mean(all_iou_scores3[:, 1]),
                np.mean(all_iou_scores4[:, 1]), np.mean(all_iou_scores5[:, 1]), np.mean(all_iou_scores_dense[:, 1])]
    sclera_ious = [np.mean(all_iou_scores[:, 2]), np.mean(all_iou_scores2[:, 2]), np.mean(all_iou_scores3[:, 2]),
                np.mean(all_iou_scores4[:, 2]), np.mean(all_iou_scores5[:, 2]), np.mean(all_iou_scores_dense[:, 2])]

    x = np.arange(len(models))  # Model indices
    width = 0.25  # Width of each bar

    # Define softer colors
    colors = {
        'pupil': '#4C72B0',  # Soft blue
        'iris': '#55A868',   # Soft green
        'sclera': '#C44E52'  # Soft red
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust size for a paper-friendly format
    ax.bar(x - width, pupil_ious, width, label='Pupil IoU', color=colors['pupil'])
    ax.bar(x, iris_ious, width, label='Iris IoU', color=colors['iris'])
    ax.bar(x + width, sclera_ious, width, label='Sclera IoU', color=colors['sclera'])

    # Add labels, title, and legend
    ax.set_xlabel('Models', fontsize=12, weight='bold')
    ax.set_ylabel('Mean IoU', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, fontsize=10)

    # Adjust legend position to avoid overlapping with bars
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=3, frameon=False)

    # Adjust grid and style
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # Ensure gridlines are behind bars

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the plot as a high-resolution image for papers
    plt.savefig('iou_comparison_plot.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
