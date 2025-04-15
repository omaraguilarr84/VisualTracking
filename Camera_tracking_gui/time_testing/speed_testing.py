import os
import time
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from torchvision import transforms
from models.opt import parse_args

# Import all models
from models.mobilenet_v1 import MobileNet2D_V1
from models.mobilenet_v2 import MobileNet2D_V2
from models.mobilenet_v3 import MobileNet2D_V3
from models.mobilenet_v4 import MobileNet2D_V4
from models.mobilenet_v5 import MobileNet2D_V5
from models.densenet_og import DenseNet2D

def load_image(image_path):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Read image with cv2 and convert to grayscale
    img = cv2.imread(image_path)
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(gray_frame)
    
    # Apply transform and add batch dimension
    image = transform(pil_image).unsqueeze(0)
    return image

def test_model_speed(model, input_image, num_iterations=100):
    model.eval()
    times = []
    
    # Warm-up run
    with torch.no_grad():
        model(input_image)
    
    # Test runs
    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            model(input_image)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time

def get_available_er_values(weights_dir, model_name):
    """Get available ER values from weight files for a specific model"""
    er_values = []
    for file in os.listdir(weights_dir):
        if file.startswith(f"TRAIN_{model_name}") and "ER" in file:
            er_value = int(file.split("ER")[1].split(".")[0])
            er_values.append(er_value)
    return sorted(er_values)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test inference speed of a specific model')
    parser.add_argument('--model', type=str, required=True, 
                      choices=['MOBILE_V1', 'MOBILE_V2', 'MOBILE_V3', 'MOBILE_V4', 'MOBILE_V5', 'DenseNet'],
                      help='Model to test')
    parser.add_argument('--er', type=int, help='Expansion ratio (only for MOBILE_V1)')
    args_test = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n=== Device Information ===")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("=" * 25 + "\n")
    
    # Load test image
    image_path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Medical_Image_Processing\demo\0.png"
    input_image = load_image(image_path).to(device)
    
    # Model configurations with weight file patterns
    weights_dir = r'C:\Users\hayde\OneDrive\Documents\Y5S2\Medical_Image_Processing\demo\time_testing\weights'
    base_models_config = {
        'MOBILE_V1': {
            'model_class': MobileNet2D_V1,
            'weight_pattern': lambda er: os.path.join(weights_dir, f'TRAIN_MOBILE_V1_ER{er}.pkl') if er is not None else os.path.join(weights_dir, 'TRAIN_MOBILE_V1.pkl')
        },
        'MOBILE_V2': {
            'model_class': MobileNet2D_V2,
            'weight_pattern': lambda er: os.path.join(weights_dir, 'TRAIN_MOBILE_V2.pkl')
        },
        'MOBILE_V3': {
            'model_class': MobileNet2D_V3,
            'weight_pattern': lambda er: os.path.join(weights_dir, 'TRAIN_MOBILE_V3.pkl')
        },
        'MOBILE_V4': {
            'model_class': MobileNet2D_V4,
            'weight_pattern': lambda er: os.path.join(weights_dir, 'TRAIN_MOBILE_V4.pkl')
        },
        'MOBILE_V5': {
            'model_class': MobileNet2D_V5,
            'weight_pattern': lambda er: os.path.join(weights_dir, 'TRAIN_MOBILE_V5.pkl')
        },
        'DenseNet': {
            'model_class': DenseNet2D,
            'weight_pattern': lambda er: os.path.join(weights_dir, 'best_model.pkl')
        }
    }
    
    # Set er value in opt if needed
    if args_test.er is not None:
        args = parse_args(er_override=args_test.er)
    
    # Get weight file path
    weight_file = base_models_config[args_test.model]['weight_pattern'](args_test.er)
    if not os.path.exists(weight_file):
        print(f"Error: Weight file {weight_file} not found")
        return
    
    # Initialize model
    model = base_models_config[args_test.model]['model_class']()
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load(weight_file, map_location=device))
    except Exception as e:
        print(f"Error loading weights from {weight_file}: {str(e)}")
        return
    
    # Test speed
    print(f"\nTesting {args_test.model}" + (f" with ER={args_test.er}" if args_test.er is not None else ""))
    avg_time, std_time = test_model_speed(model, input_image)
    
    # Print results
    print("\n=== Results ===")
    print("Model Configuration | Inference Time (ms) | Expansion Ratio | Weights")
    print("-" * 100)
    model_name = f"{args_test.model}_er{args_test.er}" if args_test.er is not None else args_test.model
    er_str = f"er={args_test.er}" if args_test.er is not None else "N/A"
    weights_file = os.path.basename(weight_file)
    print(f"{model_name:20} | {avg_time*1000:>15.2f} ± {std_time*1000:<6.2f} | {er_str:14} | {weights_file}")

if __name__ == "__main__":
    main()
