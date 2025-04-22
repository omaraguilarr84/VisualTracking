import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os



def gt_masks(path):

    ## Get label files with polygon masks
    with open(path, 'r') as f:
        data = json.load(f)
    height = data['imageHeight']
    width = data['imageWidth']

    ## Initialize empty masks
    mask_sclera = np.zeros((height, width), dtype=np.uint8)
    mask_iris = np.zeros((height, width), dtype=np.uint8)
    mask_pupil = np.zeros((height, width), dtype=np.uint8)
    mask_full_iris = np.zeros((height, width), dtype=np.uint8)
    mask_full_sclera = np.zeros((height, width), dtype=np.uint8)

    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        if shape['label'] == 'pupil':
            cv2.fillPoly(mask_pupil, [points], 255)
    
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        if shape['label'] == 'iris':
            cv2.fillPoly(mask_full_iris, [points], 255)
            mask_iris = cv2.bitwise_and(mask_full_iris, cv2.bitwise_not(mask_pupil))

    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        if shape['label'] == 'sclera':
            cv2.fillPoly(mask_full_sclera, [points], 255)
            mask_sclera = cv2.bitwise_and(mask_full_sclera, cv2.bitwise_not(mask_full_iris))

    return mask_pupil, mask_iris, mask_sclera



def main():
    path = r"C:\Users\hayde\OneDrive\Documents\Y5S2\Machine_Learning_for_Biosci\Project1_updated_021125\VisualTracking\Camera_tracking_gui\video_frames_20250413_223910\023og.json"
    mask_pupil, mask_iris, mask_sclera = gt_masks(path)

    # Display the masks
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(mask_sclera, cmap='gray')
    axs[0].set_title('Sclera Mask')
    axs[0].axis('off')

    axs[1].imshow(mask_iris, cmap='gray')
    axs[1].set_title('Iris Mask')
    axs[1].axis('off')

    axs[2].imshow(mask_pupil, cmap='gray')
    axs[2].set_title('Pupil Mask')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()