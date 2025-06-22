import tifffile
from matplotlib import pyplot as plt
import glob
import seaborn as sns
import numpy as np
file = glob.glob("metignen13-05-2025/testvncent67mbarVO2025-05-13_15-36-36/testvncent67mbarVO2025-05-13_15-36-36.ome.tif")

print(file)
reader = tifffile.imread(file)
for img in reader[60:]:
    sns.heatmap(np.minimum(img,100), cmap='nipy_spectral')
    plt.show()