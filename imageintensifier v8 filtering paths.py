import imageio.v2 as iio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
from multiprocessing import Process
import scipy
import random

def showheatmap(image):
    sns.heatmap(image, cmap='nipy_spectral')
    plt.show()
    
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
    localmaxindex = None
    topslist = []
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
    findfwhm(newrow,topslist,row, startstreamer)
    
def findstreamers(row, hor=True):
    newrow = np.zeros(len(row))
    instreamer = False
    time = 0
    for i,item in enumerate(row):
        if not instreamer:
            if item != 0:
                if time == 0:
                    startstreamer = i
                time += 1
                if time > 3:
                    instreamer = True
            elif time != 0:
                time -= 1  
        else:
            if item != 0 and time < 3:
                time += 1
            else:
                if time == 3:
                    stopstreamer = i
                time -= 1
                if time == 0:
                    instreamer = False
                    newrow[startstreamer:stopstreamer] += 3
                    if hor:
                        findtops(newrow, startstreamer, stopstreamer,row)
    else:
        if instreamer:
            newrow[startstreamer:] += 1
            # for el in np.where(row[startstreamer:]==np.max(row[startstreamer:]))[0]:
            #     newrow[el+startstreamer]+=1
    return newrow

def distancesquared(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2

def checkbranch(branch, streamer):
    for i in range(len(branch)-1):
        start = branch[i]
        stop = branch[i+1]
        som = 0
        sgn = np.sign(int(stop[0])-int(start[0]))
        if sgn == 0:
            continue
        slope = (stop[1]-start[1])/(stop[0]-start[0])
        for x in range(int(start[0]),int(stop[0]),sgn):
            if streamer[int(slope*(x-int(start[0]))+0.5)+int(start[1])][x] == 0:
                som += 1
        if som >= 10:
            return False
    return True
        

if __name__ == '__main__':
    # img = iio.imread("vincenttest_2404_1000_20_100ns_100co2_024.tif")
    img = iio.imread("vincenttest_2404_750_10_1000ns_200mbar_firstpulse5us_008.tif")
    # img = np.array([img[2*i] + img[2*i+1] for i in range(len(img)//2)])/2
    imgcopy = img.copy()[:50].flatten()
    mean = np.mean(imgcopy)
    std = np.std(imgcopy)
    newimg = np.array([[(el-mean)/std if (el-mean)/std > 0 else 0 for el in row] for row in img])
    newimg = cv2.GaussianBlur(newimg,(3,3),0)
    newerimg = np.array([[el if (newimg[i-1][j]+newimg[i-1][j-1]+newimg[i-1][j+1]+newimg[i][j-1]+newimg[i][j+1]+newimg[i+1][j]+newimg[i+1][j+1]+newimg[i+1][j-1])-max(newimg[i-1][j],newimg[i-1][j-1],newimg[i-1][j+1],newimg[i][j-1],newimg[i][j+1],newimg[i+1][j],newimg[i+1][j+1],newimg[i+1][j-1]) > 7 else 0 for j,el in enumerate(row[1:-1])] for i,row in enumerate(newimg[1:-1])])
    newestimg = np.array([findstreamers(row) for row in newerimg])
    img = img.transpose()
    # newimg = np.array([[(el-mean)/std if (el-mean)/std > 0 else 0 for el in row] for row in img])
    # newimg = cv2.GaussianBlur(newimg,(3,3),0)
    # newerimg2 = np.array([[el if (newimg[i-1][j]+newimg[i-1][j-1]+newimg[i-1][j+1]+newimg[i][j-1]+newimg[i][j+1]+newimg[i+1][j]+newimg[i+1][j+1]+newimg[i+1][j-1])-max(newimg[i-1][j],newimg[i-1][j-1],newimg[i-1][j+1],newimg[i][j-1],newimg[i][j+1],newimg[i+1][j],newimg[i+1][j+1],newimg[i+1][j-1]) > 7 else 0 for j,el in enumerate(row[1:-1])] for i,row in enumerate(newimg[1:-1])])
    newestimg2 = np.array([findstreamers(row, hor=False) for row in newerimg.transpose()]).transpose()
    betterimg = np.maximum(newestimg, newestimg2)
    
    coordsoftops = np.array([[j,i] for i,row in enumerate(betterimg) for j, el in enumerate(row) if el > 3])
    
    firsttops = dict()
    zoveel = 50
    for i in range(zoveel):
        try:
            firsttops[str(coordsoftops[i][0])] += 1
        except:
            firsttops[str(coordsoftops[i][0])] = 1
    streamerstart = int(max(firsttops, key=firsttops.get))
    if str(streamerstart+1) not in firsttops:
        firsttops[str(streamerstart+1)] = 0
    if str(streamerstart-1) not in firsttops:
        firsttops[str(streamerstart-1)] = 0
    streamerstart += (firsttops[str(int(streamerstart)+1)]-firsttops[str(int(streamerstart)-1)])/(firsttops[str(streamerstart)]+firsttops[str(int(streamerstart)+1)]+firsttops[str(int(streamerstart)-1)])
    
    angle = 0
    spinalcoords2 = []
    rigidness = 0.98
    for item in coordsoftops:
        if abs(item[0]-streamerstart) < 1 and len(spinalcoords2) == 0:
            spinalcoords2.append(item)
        elif len(spinalcoords2) > 0:
            if (item[0]-spinalcoords2[-1][0]-angle*(item[1]-spinalcoords2[-1][1]))**2+(item[1]-spinalcoords2[-1][1])**2 < 100:
                angle = rigidness*angle + (1-rigidness)*(item[0]-spinalcoords2[-1][0])
                spinalcoords2.append([((spinalcoords2[-1][0]+angle*(item[1]-spinalcoords2[-1][1]))+item[0])/2,item[1]])
                item[0] = 0
    
    branches = []
    pointsreached = []
    plt.scatter(*zip(*spinalcoords2))
    for i,branchpoint in enumerate(spinalcoords2):
        maxdistance = 10**10
        for point in coordsoftops:
            if point[1] > branchpoint[1]:
                if distancesquared(point, branchpoint) < maxdistance:
                    maxdistance = distancesquared(point, branchpoint)
                    bestpoint = point
        if maxdistance > 5000:
            continue
        angle = (bestpoint[0]-branchpoint[0])/(bestpoint[1]-branchpoint[1])
        spinalcoords = []
        itemlist = []
        rigidness = 0.98
        for item in coordsoftops:
            if distancesquared(item,bestpoint) < 1 and len(spinalcoords) == 0:
                spinalcoords.append(bestpoint)
            elif len(spinalcoords) > 0:
                if (item[0]-spinalcoords[-1][0]-angle*(item[1]-spinalcoords[-1][1]))**2 < 200 and abs(item[0]-spinalcoords[-1][0])<15:
                    angle = rigidness*angle + (1-rigidness)*(item[0]-spinalcoords[-1][0])
                    spinalcoords.append([(2*(spinalcoords[-1][0]+angle*(item[1]-spinalcoords[-1][1]))+item[0])/3,item[1]])
                    itemlist.append(tuple(item))
        if len(spinalcoords) > 20:
            if checkbranch(spinalcoords,betterimg):
                branches.append(spinalcoords)
                pointsreached.append(itemlist)
    correlations = []
    branchindexes = [[0]]
    for i in range(1, len(pointsreached)):
        inset = False
        for j in range(i):
            correlation = len(set(pointsreached[i]) & set(pointsreached[j]))/len(set(pointsreached[i])|set(pointsreached[j]))
            correlations.append([i,j, correlation])
            if not inset and correlation > 0:
                inset = True
                for branch in branchindexes:
                    if j in branch:
                        branch.append(i)
                        break
        if not inset:
            branchindexes.append([i])
    colors = list(mcolors.CSS4_COLORS.keys()) 
    random.shuffle(colors)
    for i, branch in enumerate(branchindexes):
        for el in branch:
            plt.plot(*zip(*branches[el]), color=colors[i])
    mainbranches = []
    for branch in branchindexes:
        maximumscore = 0
        mainbranch = branch[0]
        for el in branch:
            som = 0
            for el2 in branch:
                if el > el2:
                    som += correlations[int((el*(el-1))/2+el2)][2]
                elif el < el2:
                    som += correlations[int((el2*(el2-1))/2+el)][2]
            print(el,som,branchindexes.index(branch))
            if som > maximumscore:
                maximumscore = som
                mainbranch = el
        mainbranches.append(mainbranch)
    for el in mainbranches:
        plt.plot(*zip(*branches[el]), color='blue')

    plt.scatter(*zip(*coordsoftops),marker='+')
    
            
            
            
    # print(spinalcoords)
    
    # print(np.max(newimg[:100]))
    # print(np.average(newimg[:100]))
    # p = Process(target=showheatmap,args=(betterimg,))
    # p.start()
    # q = Process(target=showheatmap,args=(betterimg,))
    # q.start()
    # r = Process(target=showheatmap,args=(newerimg,))
    # r.start()
    
    # print(mean,std)

    # plt.plot(angleslist)
    # plt.plot(scipy.ndimage.gaussian_filter1d(angleslist,20))
    
    # plt.plot(newerimg[387])
    # plt.hist(np.array(newerimg[:100]).flatten(), bins=20)
    # plt.ylim([0,1000])
    plt.show()
    


