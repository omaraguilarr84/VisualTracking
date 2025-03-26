import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

impath = '/Users/omaraguilarjr/Downloads/0.png'
img = mpimg.imread(impath)

x_min, x_max, y_min, y_max = 159, 545, 166, 328
width = x_max - x_min
height = y_max - y_min

fig, ax = plt.subplots()
ax.imshow(img)
rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.axis('off')
plt.show()