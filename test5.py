import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import cv2
import tifffile
import seaborn as sns

name = "Metingen2025-06-10/fullrange67mbar6-00kV500ns2025-06-10_15-34-42/fullrange67mbar6-00kV500ns2025-06-10_15-34-42"
reader = tifffile.imread(f'{name}.ome.tif')[9]
reader = cv2.GaussianBlur(reader,(9,9),0)
# plt.plot(reader[100])
fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(reader[100])
for i in range(100,102):
    reader[i] = [100 for _ in reader[i]]
ax[0].set_ylabel("Intensity (counts)")
ax[1].imshow(reader[:300],cmap='nipy_spectral')
plt.xlabel("x-coordinate (pixels)")
plt.ylabel("y-coordiante (pixels)")

# plt.imshow(reader[:300], cmap='nipy_spectral')
# ax[1].set_aspect('equal')
plt.show()
