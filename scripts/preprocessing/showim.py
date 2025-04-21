import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np

impath = r"/Users/omaraguilarjr/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/openEDS2019/openEDS/openEDS/S_0/0.png"
impath2 = r"/Users/omaraguilarjr/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/openEDS2019/openEDS/openEDS/S_0/1.png"
img = mpimg.imread(impath)
img2 = mpimg.imread(impath2)

diff = abs(img2 - img)

threshold_value = 0.05
binary_img = np.where(img > threshold_value, 1, 0).astype(np.float32)

x_min, x_max, y_min, y_max = 159, 545, 166, 328
width = x_max - x_min
height = y_max - y_min

plt.figure()
plt.set_cmap('gray')
plt.imshow(binary_img)
plt.axis('off')
plt.show()

# plt.hist(diff)
# plt.title("Grayscale Histogram")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.show()