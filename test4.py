import tifffile
from matplotlib import pyplot as plt
import glob
import seaborn as sns
import numpy as np
file = glob.glob("Metingen2025-06-10/*/*.ome.tif")[10]

print(file)
reader = tifffile.imread(file)
for img in reader[85:]:
    sns.heatmap(np.minimum(img,100), cbar='nipy_spectral')
    plt.show()