# import warnings
# warnings.filterwarnings("ignore")

import imageio.v2 as iio
import scipy.signal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
from multiprocessing import Process, Manager
import tifffile
import time
import glob
from numba import njit
import math

def plotline(coords, title):
    plt.plot(coords)
    plt.title(title)
    plt.show()

def showheatmap(image):
    # ax = sns.heatmap(image, cmap='nipy_spectral', xticklabels=True, yticklabels=True)
    ax = sns.heatmap(image, cmap='nipy_spectral')
    ax.invert_yaxis()
    plt.show()
    
def logger(lijst, starttime):
    # print('Started')
    length = len(lijst)
    prevstringstoprint = 0
    while True:
        stringtoprint = ''
        numberofstringstoprint = 0
        started = 0
        finished = 0
        for i,item in enumerate(lijst):
            if type(item) == int:
                continue
            if item[0] == 0 or item[0] != item[1]:
                numberofstringstoprint += 1
                stringtoprint += f'{i+1}: Number of branches found: {item[0]}, checking {item[1]}'+'\n'
                started += 1
            else:
                finished += 1
        firstline = f'Finished {finished} out of {length} pictures' +'\n'
        secondline = f'Seconds busy with this set: {int(time.time()-starttime)}' + '\n'
        print('\033[1A\x1b[2K'*prevstringstoprint+firstline+secondline+stringtoprint)
        time.sleep(0.3)
        
        prevstringstoprint = numberofstringstoprint + 3
        
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
    findfwhm(newrow,topslist,row, startstreamer)
    
def findstreamers(row, hor=True):
    newrow = np.zeros(len(row))
    instreamer = False
    time = 0
    for i,item in enumerate(row):
        if not instreamer:
            if item > 5:
                if time == 0:
                    startstreamer = i
                time += 1
                if time > 3:
                    instreamer = True
            elif time != 0:
                time -= 1  
        else:
            if item > 5 and time < 3:
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
    return newrow

# def distancesquared(point1, point2):
#     return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2
# @njit(nopython=True)
def distancesquared(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

# def distanceparralel(angle, oldpoint, newpoint):
#     return (newpoint[0]-oldpoint[0])*np.sin(angle)+(newpoint[1]-oldpoint[1])*np.cos(angle)

# @njit(nopython=True)
def distanceparralel(angle, oldpoint, newpoint):
    dx = newpoint[0] - oldpoint[0]
    dy = newpoint[1] - oldpoint[1]
    return dx * math.sin(angle) + dy * math.cos(angle)


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
            try:
                if streamer[int(slope*(x-int(start[0]))+0.5)+int(start[1])][x] == 0:
                    som += 1
            except:
                som += 1
        if som >= 10:
            return False
    return True
   

def getvalue(x,y, img):
    try:
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        v1,v2,v3 = img[y1][x1],img[y1][x1+1],img[y1+1][x1]
        return v1 + (v2-v1)*(x%1)+(v3-v1)*(y%1)    
    except:
        return 0

# @njit(nopython=True)
def calculateangle(startpoint,endpoint):
    return np.arctan2(endpoint[0]-startpoint[0],endpoint[1]-startpoint[1])

def findfwhmonbranches(image,
    num,
    values,
    progresslist,
    rigidness = 0.995,
    displacementfactor = 0.1,
    maxdistmainbrach = 5,
    minimumbranchlength = 20,
    maxdistsubbranch = 5,
    maxdistancetostartbranch = 1000,
    printnumberofbranchesfound = True,
    correlationcoeff = 0.02,
    maxbend = np.pi/1.5):
    '''
    Function to find the fwhm at each point on the main and sub-branches.
    params:
    file: name of the file as str,
    rigidness: float between 0 and 1 on the stifness of the searching,
    displacementfactor: positive float for nuancedness of adding points,
    minval: float for strength of the filter,
    imagefilterblurwindow: odd int for size of window of gaussian blur. Only active when blurimg = True,
    blurimg: bool of whether to smooth the image using a gaussian filter,
    firstnlineswithoutstreamer: int of the first n lines where definitely no streamer is present. Used for statistics on background noise,
    firstntops: number of tops to determine the start x-value from,
    maxdistmainbranch: int of max distance squared, measured in pixels, allowed between two points on the main branch,
    minimumbranchlength: int of minimum number of tops a branch must contain to be considered a branch,
    maxdistsubbranch: int of max distance squared, measured in pixels, allowed between two points on the sub-branches,
    averagewidthbranch: int of the number of pixels expected as the average width of a branch,
    maxdistancetostartbranch: int of the maximum distance squared to start the search for a branch,
    printnumberofbranchesfound: bool and self explanatory
    '''
    # try:
    if True:
        start = time.time()
        rawerimage = image.copy()
        image = cv2.GaussianBlur(image,(9,9),0)
        rawimage = image.copy()
        image = np.array([findstreamers(row) for row in image])
        image2 = np.array([findstreamers(row) for row in rawimage.transpose()]).transpose()
        image = np.maximum(image,image2)
        coordsoftops = np.array([[j,i] for i,row in enumerate(image) for j, el in enumerate(row) if el > 3])
        disttocenter = [(298-el[0])**2+(13-el[1])**2 for el in coordsoftops]
        coordsoftops = [x for _,x in sorted(zip(disttocenter, coordsoftops), key=lambda pair: pair[0])]
        # generating a list of coordinates of the tops op the streamer
        finalbranches = []
        # finding the main branch
        angle = 0
        spinalcoords2 = [coordsoftops[0].copy()]
        itemlist = []
        progresslist[num] = [0,0]
        for item in coordsoftops:
            if np.sqrt(distancesquared(item, spinalcoords2[-1])) - distanceparralel(angle, spinalcoords2[-1], item) < maxdistmainbrach and abs(calculateangle(spinalcoords2[-1],item) - angle) < maxbend and np.sqrt(distancesquared(item,spinalcoords2[-1]))<20:
                angle = rigidness*angle + (1-rigidness)*calculateangle(spinalcoords2[-1],item)
                spinalcoords2.append([(displacementfactor*item[0]+spinalcoords2[-1][0]+distanceparralel(angle,spinalcoords2[-1],item)*np.sin(angle))/(displacementfactor+1),(displacementfactor*item[1]+spinalcoords2[-1][1]+distanceparralel(angle,spinalcoords2[-1],item)*np.cos(angle))/(displacementfactor+1)])
                itemlist.append(tuple(item))
                item[0] = 0
        added = []
        for item in itemlist:
            for top in coordsoftops:
                if tuple(top) not in added and distancesquared(top,item) < 10:
                    added.append(tuple(top))
                    top[0] = 0
        itemlist += added
        itemlist = [x for _,x in sorted(zip([(298-el[0])**2+(13-el[1])**2 for el in itemlist], itemlist), key=lambda pair: pair[0])]
        finalbranches.append(itemlist)
        print(time.time()-start)
        # finding the sub-branches iteratively
        for h,spine in enumerate(finalbranches):
        # for h,spine in []:
            branches = []
            pointsreached = []
            for i,branchpoint in enumerate(spine):
                maxdistance = [10**10]*8
                bestpoints = [[0,0] for _ in range(8)]
                for indexpoint,point in enumerate(coordsoftops):
                    quarant = int(calculateangle(branchpoint,point)*4/np.pi+4)
                    if quarant == 8: continue
                    if distancesquared(point, branchpoint) < maxdistance[quarant]:
                        maxdistance[quarant] = distancesquared(point, branchpoint)
                        bestpoints[quarant] = [point, indexpoint]
                for bestpointnumber, (bestpoint, indexpoint) in enumerate(bestpoints):
                    if maxdistance[bestpointnumber] > maxdistancetostartbranch:
                        continue
                    angle = calculateangle(branchpoint,bestpoint)
                    spinalcoords = [bestpoint]
                    itemlist = []
                    try:
                        for item in coordsoftops[indexpoint+1:]:
                            if item[0] != 0:
                                if np.sqrt(distancesquared(item, spinalcoords[-1])) - distanceparralel(angle, spinalcoords[-1], item) < maxdistsubbranch and abs(calculateangle(spinalcoords[-1],item) - angle) < maxbend and np.sqrt(distancesquared(item,spinalcoords[-1]))<20:
                                    angle = rigidness*angle + (1-rigidness)*calculateangle(spinalcoords[-1],item)
                                    spinalcoords.append([(displacementfactor*item[0]+spinalcoords[-1][0]+distanceparralel(angle,spinalcoords[-1],item)*np.sin(angle))/(displacementfactor+1),(displacementfactor*item[1]+spinalcoords[-1][1]+distanceparralel(angle,spinalcoords[-1],item)*np.cos(angle))/(displacementfactor+1)])
                                    itemlist.append(tuple(item))
                    except:
                        pass
                            # if (item[0]-spinalcoords[-1][0]-angle*(item[1]-spinalcoords[-1][1]))**2 < maxdistsubbranch and abs(item[0]-spinalcoords[-1][0]) < averagewidthbranch:
                            #     angle = rigidness*angle + (1-rigidness)*(item[0]-spinalcoords[-1][0])
                            #     spinalcoords.append([(displacementfactor*(spinalcoords[-1][0]+angle*(item[1]-spinalcoords[-1][1]))+item[0])/(1+displacementfactor),item[1]])
                            #     itemlist.append(tuple(item))
                    if len(spinalcoords) > minimumbranchlength:
                        if checkbranch(spinalcoords,image):
                            branches.append(itemlist)
                            pointsreached.append(itemlist)
            if len(branches) == 0:
                progresslist[num] = [len(finalbranches),h+1]
                if printnumberofbranchesfound:
                    print('\033[1A'*(num)+'\r'+f"{num}: The number of branches found is {len(finalbranches)}, checking {h+1}"+'\n'*num, end='\r')
                continue
            correlations = []
            branchindexes = [[0]]
            for i in range(1, len(pointsreached)):
                inset = False
                for j in range(i):
                    correlation = len(set(pointsreached[i]) & set(pointsreached[j]))/len(set(pointsreached[i])|set(pointsreached[j]))
                    correlations.append([i,j, correlation])
                    if not inset and correlation > correlationcoeff:
                        inset = True
                        for branch in branchindexes:
                            if j in branch:
                                branch.append(i)
                                break
                if not inset:
                    branchindexes.append([i])
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
                    if som != 0:
                        som *= len(pointsreached[el])
                    if som > maximumscore:
                        maximumscore = som
                        mainbranch = el
                    ####
                added = []
                for el in branches[mainbranch]:
                    for point in coordsoftops:
                        if tuple(point) not in added and distancesquared(el, point) < 10:
                            added.append(tuple(point))
                branches[mainbranch] += added
                branches[mainbranch] = [x for _,x in sorted(zip([(298-el[0])**2+(13-el[1])**2 for el in branches[mainbranch]], branches[mainbranch]), key=lambda pair: pair[0])]
                    ####
                mainbranches.append(mainbranch)
            
            for el in mainbranches:
                if len(branches) != 0:
                    finalbranches.append(branches[el])
                    for item in coordsoftops:
                        if tuple(item) in pointsreached[el]:
                            item[0] = 0
            progresslist[num] = [len(finalbranches),h+1]
        finalbranches = [[tuple(el) for el in branch] for branch in finalbranches]    
        values.append([num,finalbranches])
        print(time.time()-start)
    # except:
    if False:
        values.append([num,'error'])
        progresslist[num] = [1,1]
    


def finder(files):
    # reader = tifffile.imread("metignen13-05-2025/testvncent67mbarVO2025-05-13_13-33-33/testvncent67mbarVO2025-05-13_13-33-33.ome.tif")
    reader = tifffile.imread(files)[:1]
    print('\033[1A\x1b[2K', end='\r')
    # for i in range(len(reader)):
    #     print(len(reader)-1-i)
    # print('\033[1A', end='\r')
    processes = []
    values = list([])
    progresslist = list([0]*len(reader))
    for i,image in enumerate(reader):
        if np.max(np.max(image)) > 10:
            findfwhmonbranches(image,i,values,progresslist,)
        else:
            values.append([i,'error'])
    for p in processes:
        p.join()
    time.sleep(0.5)
    return values

    
if __name__ =='__main__':
    files = glob.glob('Metingen2025-05-27/*/*.ome.tif')[20:21]
    allfiles = glob.glob('*') 
    for i,file in enumerate(files):
        if len(glob.glob(f'{file[:-8]}.txt')) == 0:
            print(f'Imageset {i+1} out of {len(files)}',end='\n\n')
            values = list(finder(file))
            values.sort()
            with open(f'{file[:-8]}.txt','w') as file:
                for el in [j[1] for j in values]:
                    file.write(str(el)+'\n')
            # print('\033[1A\x1b[2K'*4,end='\r')
    # findfwhmonbranches(image)