import tifffile
from matplotlib import pyplot as plt
import cv2
img = cv2.GaussianBlur(tifffile.imread('m/grondmetingen5-73x500ns2025-05-27_14-25-10/grondmetingen5-73x500ns2025-05-27_14-25-10.ome.tif')[0],(9,9),0)
plt.imshow(img, cmap='nipy_spectral')
plt.show()