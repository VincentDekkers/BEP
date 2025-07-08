import glob
import numpy as np
from numpy import NaN
from matplotlib import pyplot as plt
import scipy.signal as ss
import cv2
import tifffile
# names = [i for i in glob.glob(
#     "Metingen2025-05-27/grondm*/*.txt")+glob.glob("Metingen2025-05-27/text*/*.txt") if "RecSettings" not in i]
names = [i for i in glob.glob(
    "Metingen2025-05-27/*/*.txt") if "RecSettings" not in i][20:]
brancheses = []
names = [x for i,x in enumerate(names) if i // 10 % 2 != 0]


def distancesquared(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2


for name in names:
    print(name)
    with open(name, 'r') as file:
        brancheses.append([eval(i[:-1]) if i != 'error\n' else []
                          for i in file.readlines()])
sidewayss = []
colors = []
left = right = 0
for branches in brancheses:
    for p,finalbranches in enumerate(branches):
        if tuple(finalbranches) == tuple([]):
            pass
        else:
            maxr = 0
            bestitem = (298, 13)
            for branch in finalbranches:
                if distancesquared((298, 13), branch[-1]) > maxr:
                    maxr = distancesquared((298, 13), branch[-1])
                    bestitem = branch[-1]
            sidewayss.append(bestitem)
            colors.append(p)
            if bestitem[0] > 298:
                right += 1
            elif bestitem[0] < 298:
                left += 1

# # Create a 2D histogram
# heatmap, xedges, yedges = np.histogram2d(*zip(*sidewayss), bins=50)

# # Plot the heatmap
# plt.imshow(heatmap.T, origin='lower', cmap='nipy_spectral', aspect='auto')
# plt.colorbar(label='Density')
# plt.title('Heatmap')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
plt.scatter(*zip(*sidewayss), c=colors, cmap='nipy_spectral', alpha=0.8)
plt.title(f"{right} to the right and {left} to the left")
ax = plt.gca()
ax.axvline(x=298, color='grey')
ax.axhline(y=13, color='gray')
plt.show()
