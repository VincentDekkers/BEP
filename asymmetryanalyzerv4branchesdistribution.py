import glob
import numpy as np
from numpy import NaN
from matplotlib import pyplot as plt
import scipy.signal as ss
import cv2
import tifffile
from matplotlib import colors
import seaborn as sns
# names = [i for i in glob.glob(
#     "Metingen2025-05-27/grondm*/*.txt")+glob.glob("Metingen2025-05-27/text*/*.txt") if "RecSettings" not in i]
names = [i for i in glob.glob(
    "Metingen2025-06-23/*/*.txt") if "RecSettings" not in i][20:]
brancheses = []
# names = [x for i,x in enumerate(names[:-10]) if i // 10 % 2 != 0]


def calculateangle(startpoint,endpoint):return np.arctan2(endpoint[0]-startpoint[0],endpoint[1]-startpoint[1])
def distancesquared(point1, point2):return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2


for name in names:
    print(name)
    with open(name, 'r') as file:
        brancheses.append([eval(i[:-1]) if i != 'error\n' else []
                          for i in file.readlines()])

xvals = []

for branches in brancheses:
    sidewaysleft = []
    sidewaysright = []
    for p,finalbranches in enumerate(branches):
        maxr = 0
        angle = 0
        # firstbranch = True
        for branch in finalbranches:
            try:
                # for _ in range(branch[0][1],branch[-1][1]):
                    xvals.append(branch[-1][0])
            except:
                pass
            # for point in branch:
        #     if distancesquared((298, 13), branch[-1]) > maxr:
        #         maxr = distancesquared((298, 13), branch[-1])
        #         # angle = np.sin(np.abs(calculateangle((298,13),branch[-1])))
        #         angle = branch[-1][0]
        # xvals.append(angle)
        # xvals[len(finalbranches)].append(int(np.abs(298-besty)))
        
        # for branch in finalbranches:
        #     if branch[-1][1] > besty - 100 and branch[-1][1] - branch[0][1] > 50:
        # xvals.append(besty)
# [print(j, len(i), np.mean(i), np.std(i)) for j,i in enumerate(xvals)]
# sns.kdeplot([j for sub in xvals[26:34] for j in sub],bw_adjust=0.5)
sns.kdeplot(np.array(xvals)*11/473-6.93, bw_adjust=0.5)

xs = np.linspace(-3,3,300)
ys = 1/(np.pi*np.sqrt(10.5-xs**2))
plt.plot(xs,ys,':',color='red')

# plt.show()
# sidewayssleft = np.sqrt(np.nanmean(np.array(sidewayssleft).transpose(),1))*11/473
# sidewayssright = np.sqrt(np.nanmean(np.array(sidewayssright).transpose(),1))*11/473

# plt.plot(list(range(100,1001,10)),sidewayssleft, c='red')
# plt.plot(list(range(100,1001,10)),sidewayssright, c='blue')


# plt.plot(list(range(100,1001,10)),sidewayssleft - sidewayssright)
# plt.gca().axhline(y=np.mean(sidewayssleft-sidewayssright), color='grey')

# plt.plot(list(range(100,1001,10)),np.array(left)/(np.array(right)+np.array(left)))
# plt.xlim([0,1000])
# plt.ylim([0,1])
# plt.xlabel(r"$\Delta t \; (\mu s)$")
plt.xlabel("sideward movement (cm)")
# plt.ylabel("Fraction of longest branches going left (cm)")
plt.gca().axvline(x=0,color='black')
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
