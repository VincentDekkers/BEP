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

colors = ['red','blue','purple','green']
# names = [i for i in glob.glob(
#     "Metingen2025-05-27/grondm*/*.txt")+glob.glob("Metingen2025-05-27/text*/*.txt") if "RecSettings" not in i]
for q in range(4):
    names = [i for i in glob.glob(
        "Metingen2025-05-27/*/*.txt") if "RecSettings" not in i][q*20:20*(q+1)]
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
                # y = 0
                for branch in finalbranches:
                    # if branch[-1][1]-13 > y:
                    #     y = branch[-1][1]-13
                    
                    dist = np.sqrt(distancesquared(branch[-1], (298, 13)))
                    if dist > maxdist:
                        maxdist = dist
                        
                        y = max([i[1] for i in branch])-13
                        totfrac = (maxdist/y)
                sideway.append(totfrac)
        sidewayss.append(sideway)
    
    reference = np.nanmean(np.array(sidewayss[:10]).flatten())
    referencestd = np.nanstd(np.array(sidewayss[:10]).flatten())
    data = np.array(sidewayss[10:]).transpose()
    averages = np.array([np.nanmean(i) for i in data])
    averages = ss.savgol_filter(averages, 15, 3)
    std = np.array([np.nanstd(i) for i in data])
    std = ss.savgol_filter(std, 15, 3)
    xvals = list(range(100, 1001, 10))
    # plt.plot(xvals, averages, color='black')
    # plt.plot(xvals, averages + std, color='black',ls=':')
    # plt.plot(xvals, averages - std, color='black',ls=':')
    # plt.xlabel(r"$\Delta t\:(\mu s)$")
    # plt.ylabel("Relative streamer width (-)")
    plt.xlim([100,1000])
    plt.xscale('log')
    plt.gca().axhline(y=reference, color=colors[q])
    # xvals2 = xvals.copy()*10
    # xvals2.sort()
    # plt.scatter(xvals2,data.flatten())
    plt.plot(xvals, averages, color=colors[q])
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


plt.xlabel(r"$\Delta t\:(\mu s)$")
plt.ylabel("Realtive streamer width (-)")
# plt.title("7.67kV 67 mbar 200 ns")
ax = plt.gca()
# ax.axhline(y=np.mean(reference)+referencestd, color='black', ls=':')
# ax.axhline(y=np.mean(reference)-referencestd, color='black', ls=':')
# ax.axhline(y=np.mean(reference), color='black')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(MultipleLocator(200))
plt.show()
