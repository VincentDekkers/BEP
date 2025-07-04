import glob
import numpy as np
from numpy import NaN
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator
import scipy.signal as ss
import cv2
import tifffile



def distancesquared(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2
referencess = []
xvalss = []
averagess = []
colors = ['red','green','blue','purple']
startvals = [33,0,45]

# names = [i for i in glob.glob(
#     "Metingen2025-05-27/grondm*/*.txt")+glob.glob("Metingen2025-05-27/text*/*.txt") if "RecSettings" not in i]
for q in range(3):
    names = [i for i in glob.glob(
        "Metingen2025-06-03/*/*.txt") if ("RecSettings" not in i) and ('1000' not in i)][startvals[q]:startvals[q]+11]
    brancheses = []
    for name in names:
        print(name)
        with open(name, 'r') as file:
            brancheses.append([eval(i[:-1]) if i != 'error\n' else [] for i in file.readlines()])
    sidewayss = []
    for i,branches in enumerate(brancheses):
        sideway = []
        for k,finalbranches in enumerate(branches):
            totfrac = 0
            totweights = 1
            if tuple(finalbranches) == tuple([]):
                sideway.append(NaN)
            else:
                maxdist = 0
                maxy = 0
                # y = 0
                for branch in finalbranches:
                    # if branch[-1][1]-13 > y:
                    #     y = branch[-1][1]-13
                    
                    dist = np.sqrt(distancesquared(branch[-1], (298, 13)))
                    if dist > maxdist:
                        maxdist = dist
                        
                        y = max([i[1] for i in branch])-13
                    # if y > maxy:
                    #     maxy = y
                sideway.append(maxdist/y)
        sidewayss.append(sideway)

    reference = np.nanmean(np.array(sidewayss[:1]).flatten())
    referencestd = np.nanstd(np.array(sidewayss[:1]).flatten())
    # data = []
    # for i in range(1,5):
    data = np.array(sidewayss[1:]).transpose()
    averages = np.array([np.nanmean(i) for i in data])
    averages = ss.savgol_filter(averages, 15, 3)
    std = np.array([np.nanstd(i) for i in data])
    std = ss.savgol_filter(std, 15, 3)
    xvals = list(range(10,1001,10))
    plt.plot(xvals,averages,color=colors[q])
    referencess.append(reference)
    # xvals = list(np.array(list(range(10,101,1)))/10)+list(range(10,101,1))+list(range(100, 1001, 10))+list(range(1000,10001,100))
plt.legend(["5.13 kV", "5.73 kV","6.70 kV"])
# plt.plot(xvals, averages, color='black')
# plt.plot(xvals, averages + std, color='black',ls=':')
# plt.plot(xvals, averages - std, color='black',ls=':')
plt.xlabel(r"$\Delta t\:(\mu s)$")
plt.ylabel("Relative sideward propagation (-)")
    # referencess.append(reference)
    # xvalss.append(xvals)
    # averagess.append(averages)
    # xvals2 = xvals.copy()*10
    # xvals2.sort()
    # plt.scatter(xvals2,data.flatten())
    # plt.plot(xvals, averages, color=colors[q])
    # plt.plot(xvals, averages + std, ':', color=colors[q])
    # plt.plot(xvals, averages - std, ':', color=colors[q])


    # data = np.array(sidewayss[11:]).transpose()
    # averages = np.array([np.nanmean(i) for i in data])
    # averages = ss.savgol_filter(averages, 15, 2)
    # std = np.array([np.nanstd(i) for i in data])
    # std = ss.savgol_filter(std, 15, 2)
    # xvals = list(range(10, 101, 1))
    # plt.plot(xvals, averages, color='blue')
    # plt.plot(xvals, averages + std, 'b:')
    # plt.plot(xvals, averages - std, 'b:')
# plt.legend(["3.76 kV","7.00 kV","7.76 kV", "5.73 kV"])
# for q,reference in enumerate(referencess):
#     plt.gca().axhline(y=reference, color=colors[q])
#     plt.plot()
plt.xlim([30,1000])
plt.xscale('log')
# plt.xlabel(r"$\Delta t\:(\mu s)$")
# plt.ylabel("Realtive sideward propagation (-)")

# plt.title("7.67kV 67 mbar 200 ns")
ax = plt.gca()
for i in range(3):
    # ax.axhline(y=np.mean(referencess[i])+referencestd, color='black', ls=':')
    # ax.axhline(y=np.mean(referencess)-referencestd, color='black', ls=':')
    ax.axhline(y=np.mean(referencess[i]), color=colors[i])
ax.xaxis.set_major_formatter(ScalarFormatter())
# ax.xaxis.set_minor_formatter(ScalarFormatter())
# ax.xaxis.set_minor_locator(MultipleLocator(200))
plt.show()
