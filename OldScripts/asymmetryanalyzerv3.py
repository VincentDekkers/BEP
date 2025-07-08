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
names = [x for i,x in enumerate(names[:-10]) if i // 10 % 2 != 1]


def distancesquared(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2


for name in names:
    print(name)
    with open(name, 'r') as file:
        brancheses.append([eval(i[:-1]) if i != 'error\n' else []
                          for i in file.readlines()])
sidewayssleft = []
sidewayssright = []
# colors = []
left = [0]*91
right = [0]*91
for branches in brancheses:
    sidewaysleft = []
    sidewaysright = []
    for p,finalbranches in enumerate(branches):
        if tuple(finalbranches) == tuple([]):
            sidewaysleft.append(NaN)
            sidewaysright.append(NaN)
        else:
            maxrleft = 0
            maxrright = 0
            bestitemleft = (298, 13)
            bestitemright = (298, 13)
            # firstbranch = True
            for branch in finalbranches:
                # for point in branch:
                if branch[-1][0] < 298:
                    if distancesquared((298, 13), branch[-1]) > maxrleft:
                        maxrleft = distancesquared((298, 13), branch[-1])
                        bestitemleft = branch[-1][0]
                if branch[-1][0] > 298:
                    if distancesquared((298, 13), branch[-1]) > maxrright:
                        maxrright = distancesquared((298, 13), branch[-1])
                        bestitemright = branch[-1][0]
            if bestitemleft != (298,13):
                sidewaysleft.append(maxrleft)
            else:
                sidewaysleft.append(NaN)
            if bestitemright != (298,13):
                sidewaysright.append(maxrright)
            else:
                sidewaysright.append(NaN)
            if maxrleft > maxrright:
                left[p] += 1
            elif maxrright > maxrleft:
                right[p] += 1
    sidewayssleft.append(sidewaysleft)
    sidewayssright.append(sidewaysright)


sidewayssleft = np.sqrt(np.nanmean(np.array(sidewayssleft).transpose()))*11/473
sidewayssright = np.sqrt(np.nanmean(np.array(sidewayssright).transpose()))*11/473
print(np.mean(sidewayssleft), np.mean(sidewayssright))
# plt.plot(list(range(100,1001,10)),sidewayssleft, c='red')
# plt.plot(list(range(100,1001,10)),sidewayssright, c='blue')


# plt.plot(list(range(100,1001,10)),sidewayssleft - sidewayssright)
# plt.gca().axhline(y=np.mean(sidewayssleft-sidewayssright), color='grey')
print(np.sum(left), np.sum(right))
plt.plot(list(range(100,1001,10)),ss.savgol_filter(np.array(left)/(np.array(right)+np.array(left)),50,2))

plt.xlim([0,1000])
plt.ylim([0,1])
plt.xlabel(r"$\Delta t \; (\mu s)$")
plt.ylabel("Fraction of longest branches going left (cm)")
plt.gca().axhline(y=0.5,color='black')
plt.show()


            # colors.append(p)
                    # sidewayss.append(point)
                    # # colors.append(p)
                    # if point[0] > 298:
                    #     right += 1
                    # elif point[0] < 298:
                    #     left += 1
                    # if firstbranch:
            # sidewayss.append(bestitem)
            # if bestitem[0] > 298:
            #     right[p] += 1
            # elif bestitem[0] < 298:
            #     left[p] += 1
                    #     if point[0] > 298:
                    #         right += 1
                    #     elif point[0] < 298:
                    #         left += 1
                       
                # firstbranch = False


# sidewayss.append((0,0))
# sidewayss.append((600,550))
# maxx = max(list(zip(*sidewayss))[0])-min(list(zip(*sidewayss))[0])
# maxy = max(list(zip(*sidewayss))[1])-min(list(zip(*sidewayss))[1])
# heatmap, xedges, yedges = np.histogram2d(*zip(*sidewayss), bins=(maxx,maxy))
# heatmap = np.log(np.maximum(1,heatmap))


# names = [i for i in glob.glob(
#     "Metingen2025-05-27/*/*.txt") if "RecSettings" not in i][20:]
# brancheses = []
# names = [x for i,x in enumerate(names[:-10]) if i // 10 % 2 != 0]


# def distancesquared(point1, point2):
#     return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2


# for name in names:
#     print(name)
#     with open(name, 'r') as file:
#         brancheses.append([eval(i[:-1]) if i != 'error\n' else []
#                           for i in file.readlines()])
# sidewayss = []
# # colors = []
# left = right = 0
# for branches in brancheses:
#     for p,finalbranches in enumerate(branches):
#         if tuple(finalbranches) == tuple([]):
#             pass
#         else:
#             maxr = 0
#             bestitem = (298, 13)
#             firstbranch = True
#             for branch in finalbranches:
#                 for point in branch:
#                 # if distancesquared((298, 13), branch[-1]) > maxr:
#                 #     maxr = distancesquared((298, 13), branch[-1])
#                     # bestitem = point
#                     sidewayss.append(point)
#                     # colors.append(p)
#                     if point[0] > 298:
#                         right += 1
#                     elif point[0] < 298:
#                         left += 1
#                     if firstbranch:
#                         sidewayss.append(point)
#                         if point[0] > 298:
#                             right += 1
#                         elif point[0] < 298:
#                             left += 1
                            
#                 firstbranch = False


# sidewayss.append((0,0))
# sidewayss.append((600,550))
# maxx = max(list(zip(*sidewayss))[0])-min(list(zip(*sidewayss))[0])
# maxy = max(list(zip(*sidewayss))[1])-min(list(zip(*sidewayss))[1])
# heatmap2, xedges, yedges = np.histogram2d(*zip(*sidewayss), bins=(maxx,maxy))
# heatmap2 = np.log(np.maximum(1,heatmap2))
# heatmap = heatmap - heatmap2

# # Plot the heatmap
# plt.imshow(heatmap.T, origin='lower', cmap='nipy_spectral', aspect='auto', norm=colors.LogNorm())
# plt.colorbar(label='Branch density')
# plt.title(f"{right} to the right and {left} to the left")
# plt.gca().invert_yaxis()
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
# plt.scatter(*zip(*sidewayss), c=colors, cmap='nipy_spectral', alpha=0.5)
# plt.title(f"{sum(right)} to the right and {sum(left)} to the left")
# ax = plt.gca()
# ax.axvline(x=298, color='grey')
# ax.axhline(y=(1497)/(1497+1219), color='gray')
# plt.show()
# plt.scatter(list(range(100,1001,10)),np.array(left)/(np.array(right)+np.array(left)))
# plt.xlabel(r"$\Delta t \; (\mu s)$")
# plt.ylabel("Fraction of longest branches going left (-)")
# plt.xlim([0,1000])
# plt.show()
