import glob
import numpy as np
from numpy import NaN
from matplotlib import pyplot as plt
import scipy.signal as ss
import cv2
import tifffile
from matplotlib import colors
# names = [i for i in glob.glob(
#     "Metingen2025-05-27/grondm*/*.txt")+glob.glob("Metingen2025-05-27/text*/*.txt") if "RecSettings" not in i]
names = [i for i in glob.glob(
    "Metingen2025-05-27/*/*.txt") if "RecSettings" not in i][20:]
brancheses = []
names = [x for i,x in enumerate(names[:-10]) if i // 10 % 2 != 0]


def distancesquared(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2


for name in names:
    print(name)
    with open(name, 'r') as file:
        brancheses.append([eval(i[:-1]) if i != 'error\n' else []
                          for i in file.readlines()])
sidewayss = []
# colors = []
left = right = 0
for branches in brancheses:
    for p,finalbranches in enumerate(branches):
        if tuple(finalbranches) == tuple([]):
            pass
        else:
            maxr = 0
            bestitem = (298, 13)
            for branch in finalbranches:
                for point in branch:
                # if distancesquared((298, 13), branch[-1]) > maxr:
                #     maxr = distancesquared((298, 13), branch[-1])
                    # bestitem = point
                    sidewayss.append(point)
                    # colors.append(p)
                    if point[0] > 298:
                        right += 1
                    elif point[0] < 298:
                        left += 1


sidewayss.append((0,0))
sidewayss.append((600,550))
maxx = max(list(zip(*sidewayss))[0])-min(list(zip(*sidewayss))[0])
maxy = max(list(zip(*sidewayss))[1])-min(list(zip(*sidewayss))[1])
heatmap, xedges, yedges = np.histogram2d(*zip(*sidewayss), bins=(maxx,maxy))

# Plot the heatmap
plt.imshow(heatmap.T, origin='lower', cmap='nipy_spectral', aspect='auto', norm=colors.LogNorm())
plt.colorbar(label='Density')
plt.title(f"{right} to the right and {left} to the left")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
# plt.scatter(*zip(*sidewayss), alpha=0.01)
# plt.title(f"{right} to the right and {left} to the left")
# ax = plt.gca()
# ax.axvline(x=298, color='grey')
# ax.axhline(y=13, color='gray')
# plt.show()
