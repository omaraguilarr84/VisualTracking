import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

impath = r"C:\Users\omarh\OneDrive - Georgia Institute of Technology\openEDS2019\openEDS\openEDS\S_0\0.png"
impath2 = r"C:\Users\omarh\OneDrive - Georgia Institute of Technology\openEDS2019\openEDS\openEDS\S_0\1.png"
img = mpimg.imread(impath)
img2 = mpimg.imread(impath2)

diff = img2 - img

x_min, x_max, y_min, y_max = 159, 545, 166, 328
width = x_max - x_min
height = y_max - y_min

plt.figure()
plt.set_cmap('gray')
plt.imshow(diff)
plt.axis('off')
plt.show()