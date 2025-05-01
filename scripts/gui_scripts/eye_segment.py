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
import datetime
import argparse

def init_model(model_path, device):
    try:
        print(f"Using device: {device}")
        # Create model instance
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
    pred_img = pred_map.cpu().numpy() / 3.0
    pred_img = pred_img.squeeze(0)

    # Normalize the image to the range [0, 255]
    normalized_image = (pred_img - np.min(pred_img)) / (np.max(pred_img) - np.min(pred_img)) * 255
    # Convert to uint8 type
    normalized_image = normalized_image.astype(np.uint8)
    return normalized_image

def segment_image(file_path):
    # Initialize the model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'TRAIN_MOBILE_V1.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(model_path, device)

    # Load the image
    if file_path.endswith('.jpg') or file_path.endswith('.jpeg') or file_path.endswith('.png'):
        # Read the image and run inference
        image = cv2.imread(file_path)
        pred_img = process_image(image, model, device) 
        # Save the predicted image
        base_path = os.path.splitext(file_path)[0]
        new_file_path = f"{base_path}_predicted.png"
        cv2.imwrite(new_file_path, pred_img)
        print(f"Predicted image saved at {new_file_path}")
        return pred_img
    elif os.path.isdir(file_path):
        for file in os.listdir(file_path):
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                image = cv2.imread(os.path.join(file_path, file))
                pred_img = process_image(image, model, device)
                base_path = os.path.splitext(file)[0]
                new_file_path = f"{file_path}/{base_path}_predicted.png"
                cv2.imwrite(new_file_path, pred_img)
                print(f"Predicted image saved at {new_file_path}")
            else:
                print(f"Skipping file: {file}") 
        return
    else:
        print("Error: Invalid file format")
        return
    
    


