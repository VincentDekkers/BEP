import glob
import numpy as np
from numpy import NaN
from matplotlib import pyplot as plt
import scipy.signal as ss
import cv2
import tifffile
from matplotlib import colors
import matplotlib.cm as cmx
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
sidewayss = [[] for _ in range(91)]
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
                    sidewayss[p].append(point)
                    # colors.append(p)
                    if point[0] > 298:
                        right += 1
                    elif point[0] < 298:
                        left += 1


first = True
for sideway in sidewayss:
    # Create a 2D histogram
    sideway.append((0,0))
    sideway.append((600,550))
    maxx = max(list(zip(*sideway))[0])-min(list(zip(*sideway))[0])
    maxy = max(list(zip(*sideway))[1])-min(list(zip(*sideway))[1])
    heatmap, xedges, yedges = np.histogram2d(*zip(*sideway), bins=(maxx,maxy))
    heatmap = np.log(heatmap.T)
    if first:
        first = False
        maxxx = np.max(heatmap)
        w,h = heatmap.shape
        out = cv2.VideoWriter("totstreamers2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (h,w))
        cNorm = colors.Normalize(vmin=0, vmax=maxxx)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='nipy_spectral')
    values = scalarMap.to_rgba(heatmap)[:,:,:3]
    values *= 255
    values = values.astype(np.uint8)
    for _ in range(2):
        out.write(values) # frame is a numpy.ndarray with shape (1280, 720, 3)
    # Plot the heatmap
    # plt.imshow(heatmap.T, origin='lower', cmap='nipy_spectral', aspect='auto', norm=colors.LogNorm())
    # plt.colorbar(label='Density')
    # plt.title(f"{right} to the right and {left} to the left")
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.show()
out.release()
# plt.scatter(*zip(*sidewayss), alpha=0.01)
# plt.title(f"{right} to the right and {left} to the left")
# ax = plt.gca()
# ax.axvline(x=298, color='grey')
# ax.axhline(y=13, color='gray')
# plt.show()
