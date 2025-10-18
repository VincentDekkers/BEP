import numpy as np
from matplotlib import pyplot as plt
import glob
import tifffile
import seaborn as sns

file = glob.glob("m/*/*.ome.tif")[0]
reader = tifffile.imread(file)
for img in reader:
    sns.heatmap(np.minimum(img,300), cmap='nipy_spectral',cbar=False)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.show()
