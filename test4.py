import tifffile
from matplotlib import pyplot as plt
import glob
import seaborn as sns
import numpy as np
fig,ax = plt.subplots(1,3)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')
i = 0
file = glob.glob("Metingen2025-06-10/fullrange67mbar6-00kV500ns2025-06-10_15-44-25/fullrange67mbar6-00kV500ns2025-06-10_15-44-25.ome.tif")
reader = tifffile.imread(file)
for img in reader[6:7]:
    sns.heatmap(np.minimum(img,100), cmap='nipy_spectral',cbar=False, ax=ax[i])
    ax[i].axis('off')
    ax[i].text(200,500,r"stage I, 1.6 $\mu$s",color='white', size=30)
    
    
i = 1
file = glob.glob("Metingen2025-06-10/f*/*.tif")[35]
reader = tifffile.imread(file)
for img in reader[33:34]:
    sns.heatmap(np.minimum(img,100), cmap='nipy_spectral',cbar=False, ax=ax[i])
    ax[i].axis('off')
    ax[i].text(200,500,r"stage III, 430 $\mu$s",color='white', size=30)
    
    
i = 2
file = glob.glob("Metingen2025-06-10/f*/*.tif")[45]
reader = tifffile.imread(file)
for img in reader[71:72]:
    sns.heatmap(np.minimum(img,100), cmap='nipy_spectral',cbar=False, ax=ax[i])
    ax[i].axis('off')
    ax[i].text(200,500,r"stage V, 8.1 ms",color='white', size=30)
    plt.show()