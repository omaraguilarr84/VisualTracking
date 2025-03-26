import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Image dimensions for reference
IMG_HEIGHT = 400
IMG_WIDTH = 640

# Fixed bounding box dimensions
RECT_WIDTH = 450
RECT_HEIGHT = 200

def roundness(contour):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)

def find_largest_circular_component(thresh):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh)
    max_area = 0
    pupil_center = None
    best_roundness = 0
    best_bbox = None

    for label in range(1, num_labels):  # skip background
        component_mask = (labels == label).astype(np.uint8)
        contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = contours[0]
        r = roundness(contour)
        area = cv.contourArea(contour)

        if r > 0.6 and area > max_area:
            max_area = area
            best_roundness = r
            pupil_center = tuple(map(int, centroids[label]))
            best_bbox = stats[label].tolist()  # convert to list for unpacking

    return pupil_center, best_bbox

def show_image(title, image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_histogram(image, title):
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def main(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
    show_image('Original Image', img)

    # Mask out edges
    mask = np.ones_like(img, dtype=np.uint8)
    mask[:, :100] = 0
    mask[:, -100:] = 0
    masked_img = cv.bitwise_and(img, img, mask=mask)
    show_image('Edge Masked Image', masked_img)

    # Histogram equalization
    equalized = cv.equalizeHist(masked_img)
    show_image('Histogram Equalized Image', equalized)

    # Show histogram to analyze pixel intensities
    show_histogram(equalized, 'Histogram of Equalized Image')

    # Try Otsu's thresholding
    _, thresh_otsu = cv.threshold(equalized, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    show_image("Thresholded Image (Otsu's Method)", thresh_otsu)

    # Improved morphological cleanup: closing then opening with larger kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    closed = cv.morphologyEx(thresh_otsu, cv.MORPH_CLOSE, kernel)
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
    show_image('Cleaned Image (Close + Open)', opened)

    # Find the largest connected component with high roundness (likely pupil)
    pupil_center, bbox = find_largest_circular_component(opened)

    if pupil_center is None or bbox is None:
        print("Pupil not found.")
        return

    x, y, w, h = bbox[:4]  # ensure only the first 4 values are unpacked

    # Plotting the image and bounding box from connected component
    plt.imshow(img, cmap='gray')
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title('Detected Pupil Region')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    img_path = '/Users/omaraguilarjr/Downloads/0.png'
    main(img_path=img_path)
