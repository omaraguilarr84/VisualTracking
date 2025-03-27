import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
path = '/Users/omaraguilarjr/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/openEDS2019/openEDS/openEDS/S_0/150.npy'
data = np.load(path)

# Display as image
plt.imshow(data, cmap='gray')  # Use 'gray' for grayscale, remove it for color
plt.axis('off')                # Optional: hide axes
plt.show()