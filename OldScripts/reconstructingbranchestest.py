import glob
import numpy as np
from numpy import NaN
from matplotlib import pyplot as plt
import scipy.signal as ss
import cv2
import tifffile

def findfwhm(newrow, topslist, row, startstreamer):
    for indextop in topslist:
        height = row[indextop]
        i = j = indextop 
        lefthalf = righthalf = False
        try:
            while True:
                i += 1
                if i in topslist:
                    righthalf = True
                if row[i] < height/2:
                    break
            while True:
                j -= 1
                if j in topslist:
                    lefthalf = True
                if row[j]<height/2:
                    break
            if not righthalf and not lefthalf:
                newrow[indextop] += i - j
            elif not righthalf and lefthalf:
                newrow[indextop] += 2*(i - indextop)
            elif righthalf and not lefthalf:
                newrow[indextop] += 2*(indextop - j)
            else:
                newrow[indextop] += 1 # this is a placeholder, TODO: Build interpolation alg.
        except:
            newrow[indextop] += 0

        
def findtops(newrow, startstreamer, stopstreamer, row):
    localmax = 1
    localmaxindex = 0
    topslist = []
    printed = False
    for i,item in enumerate(row[startstreamer:stopstreamer]):
        if item > localmax:
            localmax = item
            localmaxindex = i
            printed = False
        elif item < localmax - 1:
            localmax = item + 1
            if not printed:
                topslist.append(localmaxindex + startstreamer)
                printed=True
                newrow[localmaxindex+startstreamer] += 1
    findfwhm(newrow,topslist,row, startstreamer)
    

def distancesquared(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2

def getvalue(x,y, img):
    try:
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        v1,v2,v3 = img[y1][x1],img[y1][x1+1],img[y1+1][x1]
        return v1 + (v2-v1)*(x%1)+(v3-v1)*(y%1)    
    except:
        return 0

# names = [i for i in glob.glob(
#     "Metingen2025-05-27/grondm*/*.txt")+glob.glob("Metingen2025-05-27/text*/*.txt") if "RecSettings" not in i]
names = [i for i in glob.glob(
    "Metingen2025-06-10/te*/*.txt") if ("RecSettings" not in i) and ("1000") not in i]
brancheses = []
for name in names:
    print(name)
    with open(name, 'r') as file:
        brancheses.append([eval(i[:-1]) if i != 'error\n' else [] for i in file.readlines()])
sidewayss = []
for i,branches in enumerate(brancheses):
    image = tifffile.imread(f"{names[i][:-4]}.ome.tif")
    sideway = []
    for k,finalbranches in enumerate(branches):
        rawimage = cv2.GaussianBlur(image[k],(9,9),0)
        totfrac = 0
        totweights = 1
        if tuple(finalbranches) == tuple([]):
            sideway.append(NaN)
        else:
            maxdist = 0
            for branch in finalbranches:
                
                # start = -30
                # stop = -6
                # x1,y1 = branch[start]
                # x2,y2 = branch[stop]
                # ystep = (x2-x1)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
                # xstep = (y1-y2)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
                # xm = (x2+x1)/2
                # ym = (y2+y1)/2
                # points = []
                # for j in range(-100,100):
                #     points.append(getvalue(xm+j*xstep,ym+j*ystep,rawimage))
                # topslist = np.zeros(len(points))
                # findtops(topslist, 0, -1, points)
                # weight = max(topslist[len(points)//2-10:len(points)//2+10])
                # weight = 1
                
                dist = np.sqrt(distancesquared(branch[-1], (298, 13)))
                if dist > maxdist:
                    maxdist = dist
                    
                    y = max([i[1] for i in branch])-13
                    totfrac = (dist/y)
            sideway.append(totfrac)
    sidewayss.append(sideway)

reference = np.nanmean(np.array(sidewayss[:10]).flatten())
referencestd = np.nanstd(np.array(sidewayss[:10]).flatten())
data = np.array(sidewayss[10:]).transpose()
averages = np.array([np.nanmean(i) for i in data])
averages = ss.savgol_filter(averages, 20, 3)
std = np.array([np.nanstd(i) for i in data])
std = ss.savgol_filter(std, 20, 3)
xvals = list(range(100, 1001, 10))
xvals2 = xvals.copy()*10
xvals2.sort()
plt.scatter(xvals2,data.flatten())
plt.plot(xvals, averages, color='blue')
plt.plot(xvals, averages + std, 'b:')
plt.plot(xvals, averages - std, 'b:')


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
plt.ylabel("Ratio radius travelled/vertical distance travelled.")
# plt.title("7.67kV 67 mbar 200 ns")
ax = plt.gca()
ax.axhline(y=np.mean(reference)+referencestd, color='blue', ls=':')
ax.axhline(y=np.mean(reference)-referencestd, color='blue', ls=':')
ax.axhline(y=np.mean(reference), color='blue')
plt.show()
