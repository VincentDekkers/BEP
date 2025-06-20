import imageio.v2 as iio
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from multiprocessing import Process

img = np.array(iio.imread("flatfield_2404_1000exps.tif"), dtype='float64')
print(np.max(img), np.min(img))
# sns.heatmap(img, cmap='nipy_spectral')
# plt.show()