import tifffile
from matplotlib import pyplot as plt
import glob
import seaborn as sns
import numpy as np
file = glob.glob("Metingen2025-06-23/twentymbar3-76vnitrogen2025-06-23_11-28-03/twentymbar3-76vnitrogen2025-06-23_11-28-03.ome.tif")

print(file)
reader = tifffile.imread(file)
for img in reader[60:]:
    sns.heatmap(np.minimum(img,100), cmap='nipy_spectral')
    plt.show()