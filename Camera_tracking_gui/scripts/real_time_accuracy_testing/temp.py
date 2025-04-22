import os
import shutil

def collect_json_and_images(source_dir, output_dir=r'C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\real_time_accuracy_testing\real_time_ds'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Walk through the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(source_dir, filename)
            base_name = os.path.splitext(filename)[0]

            # Try to find the corresponding image file
            for ext in image_extensions:
                image_path = os.path.join(source_dir, base_name + ext)
                if os.path.exists(image_path):
                    # Copy both files to output directory
                    shutil.copy(json_path, os.path.join(output_dir, base_name+'2'+'.json'))
                    shutil.copy(image_path, os.path.join(output_dir, base_name+'2' + ext))
                    print(f"Copied: {filename} and {base_name + ext}")
                    break
            else:
                print(f"⚠️ Image for {filename} not found.")

if __name__ == '__main__':
    # Example: replace with your actual folder
    source_folder = r'C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\video_frames_20250414_115505'
    collect_json_and_images(source_folder)