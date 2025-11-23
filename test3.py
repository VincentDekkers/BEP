import numpy as np
from matplotlib import pyplot as plt
import glob
import tifffile
import seaborn as sns
import cv2

file = glob.glob("m/*/*.ome.tif")[0]
reader = tifffile.imread(file)
for img in reader:
    img = cv2.GaussianBlur(img,(9,9),0)
    sns.heatmap(np.minimum(img,100), cmap='nipy_spectral',cbar=True)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.show()
