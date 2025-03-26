import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
path = '/Users/omaraguilarjr/Downloads/1.npy'
data = np.load(path)

# Display as image
plt.imshow(data, cmap='gray')  # Use 'gray' for grayscale, remove it for color
plt.axis('off')                # Optional: hide axes
plt.show()