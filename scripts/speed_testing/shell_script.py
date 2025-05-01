import os

# List of models to test
models = [
    'MOBILE_V1', 'MOBILE_V2', 'MOBILE_V3', 'MOBILE_V4', 'MOBILE_V5', 'DenseNet',
    'MOBILE_V1_AP', 'MOBILE_V2_AP', 'MOBILE_V3_AP', 'MOBILE_V4_AP', 'MOBILE_V5_AP', 'DenseNet_AP'
]

# Optional expansion ratios for MOBILE_V1

# Path to the speed_testing.py script
script_path = r"c:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\scripts\time_testing\speed_testing.py"

# Run the script for each model
for model in models:
    command = f"python {script_path} --model {model}"
    os.system(command)