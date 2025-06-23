import glob
import numpy as np
from numpy import NaN
from matplotlib import pyplot as plt
import scipy.signal as ss


def distancesquared(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2

names = [i for i in glob.glob("Metingen2025-06-23/*/*.txt") if "RecSettings" not in i]
# names = [i for i in glob.glob(
    # "Metingen2025-05-27/grondm*/*.txt")+glob.glob("Metingen2025-05-27/text*/*.txt") if "RecSettings" not in i]
brancheses = []
for name in names:
    print(name)
    with open(name, 'r') as file:
        brancheses.append([eval(i[:-1]) for i in file.readlines()])
sidewayss = []
for branches in brancheses:
    sideway = []
    for finalbranches in branches:
        maxy = 0
        maxr = 0
        if tuple(finalbranches) == tuple([]):
            sideway.append(NaN)
        else:
            for i, branch in enumerate(finalbranches):
                ys = [el[1]-13 for el in branch[:-1]]
                rs = [np.sqrt(distancesquared(el, (298, 13)))
                      for el in branch[:-1]]
                if max(ys) > maxy:
                    maxy = max(ys)
                if max(rs) > maxr:
                    maxr = max(rs)
            sideway.append(maxr/maxy)
    sidewayss.append(sideway)
# sidewayss = np.arccos(1/np.array(sidewayss))*180/np.pi
# sidewayss = np.array(sidewayss)*(11/473)
reference = np.array(sidewayss[:10]).flatten()
referencestd = reference.std()
reference = reference.mean()
data = []
xvals = []
for i in range(1,3):
    data += list(np.array(sidewayss[10*i:10*(i+1)]).transpose())
    xvals += list(np.array(list(range(10**(i+1), 10**(i+2)+1, 10**(i)))))
averages = np.array([np.nanmean(i) for i in data])
averages = ss.savgol_filter(averages, 15, 2)
std = np.array([np.nanstd(i) for i in data])
std = ss.savgol_filter(std, 15, 2)
plt.plot(xvals, averages, color='black')
plt.plot(xvals, averages + std, color='black',ls=':')
plt.plot(xvals, averages - std, color='black',ls=':')
plt.xlabel(r"$\Delta t\:(\mu s)$")
plt.ylabel("Relative streamer width (-)")
# plt.title(r"Relative uncertainty of relative width of streamers")
plt.xscale('log')
plt.xlim([100,10000])
ax = plt.gca()
ax.axhline(y=np.mean(reference)+referencestd, color='blue', ls=':')
ax.axhline(y=np.mean(reference)-referencestd, color='blue', ls=':')
ax.axhline(y=np.mean(reference), color='blue')
plt.show()
