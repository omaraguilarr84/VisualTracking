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

def init_model(model_path, device):
    try:
        print(f"Using device: {device}")
        # Create model instance
        model = DenseNet2D(dropout=True,prob=0.2)
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


def main():

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(r'C:\Users\hayde\OneDrive\Documents\Y5S2\Medical_Image_Processing\demo\best_model.pkl', device)

    # Initialize webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    print("Webcam opened successfully. Press 'q' to quit.")
    
    # Initialize variables for FPS calculation
    prev_time = 0
    fps = 0
    
    # Display size
    display_width = 640 
    display_height = 480

    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Video recording settings
    record_video = False
    recording_start_time = 0
    recording_duration = 45
    frame_count = 0
    output_folder = None
    video_writer = None
    
    print(f"Press 'r' to start recording a {recording_duration}-second video")
    print(f"Press 'q' to quit")

    #############
    # Main loop #
    #############
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # prepare frame for model
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray_frame)
        img_tensor = transform(pil_image)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)
        
        # Get prediction map using utils.get_predictions
        pred_map = utils.get_predictions(output)
        pred_img = pred_map.cpu().numpy() / 3.0  # Scale to [0, 1] range
        pred_img = pred_img.squeeze(0)
        
        pred_bgr = cv2.cvtColor((pred_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Resize original frame and prediction to display size
        frame_display = cv2.resize(frame, (display_width, display_height))
        pred_display = cv2.resize(pred_bgr, (display_width, display_height))
        
        # Concatenate original and prediction horizontally
        combined_frame = np.hstack((frame_display, pred_display))
        
        # Add FPS text to the frame
        cv2.putText(combined_frame, f'FPS: {fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add labels for each image
        cv2.putText(combined_frame, 'Original', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(combined_frame, 'Prediction', (display_width + 10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        #####################
        # Recording Actions #
        #####################
        if record_video:
            elapsed_time = current_time - recording_start_time
            remaining_time = recording_duration - elapsed_time
            if remaining_time > 0:
                cv2.putText(combined_frame, f'Recording: {remaining_time:.1f}s', (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Write frame to video file
                if video_writer is not None:
                    video_writer.write(combined_frame)
                
                ## Comment next line out to enable saving frames
                output_folder = False
                # Save original and prediction frames
                if output_folder:
                    # Format frame number with leading zeros
                    frame_str = f"{frame_count:03d}"
                    
                    # Save original frame
                    og_path = os.path.join(output_folder, f"{frame_str}og.jpg")
                    cv2.imwrite(og_path, frame_display)
                    
                    # Save prediction frame
                    pred_path = os.path.join(output_folder, f"{frame_str}pred.jpg")
                    cv2.imwrite(pred_path, pred_display)
                    
                    frame_count += 1
            else:
                # Stop recording after the specified duration
                record_video = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print(f"Recording completed. Video saved to {output_folder}.mp4")
        
        # Display the combined frame
        cv2.imshow('Webcam Feed and Prediction', combined_frame)
        
        #########################
        # Handle keyboard input #
        #########################
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and not record_video:
            # Create timestamp-based folder name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = f"video_frames_{timestamp}"
            
            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f"{output_folder}.mp4"
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (display_width*2, display_height))
            
            recording_start_time = current_time
            record_video = True
            frame_count = 0
            print(f"Recording started. Will save video to {video_path}")
    
    # Release the webcam and close windows
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 