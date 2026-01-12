import imageio.v2 as iio
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from multiprocessing import Process, Manager
import tifffile
from numpy import nan
import yaml

def distancesquared(coord1, coord2): # Function to calculate the distance squared between two points
    return (coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2

def findstart(twig, maxdist, front): # As some of the first tops are quite far from the rest of the top, these are excluded from the calculation
    if not front:
        front = -1
    for i in range(1,len(twig)-1):
        if distancesquared(twig[front*i], twig[front*(i+1)]) < maxdist:
            return i
    else:
        return nan
    
def getvalue(x,y, img): # linear interpolation to get sub-pixel values
    try:
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        v1,v2,v3,v4 = np.array([img[y1][x1],img[y1][x1+1],img[y1+1][x1],img[y1+1][x1+1]],dtype=float)
        xdiff = x-x1
        ydiff = y-y1
        return v1 + (v2-v1)*(xdiff)+(v3-v1)*(ydiff)
    except:
        return 0
    
def quadraticfwhmextrapolation(indextop,row):
    height = row[indextop]
    i = j = indextop
    previ = prevj = height
    while True:
        i += 1
        if row[i] > previ:
            break
        previ = row[i]
    while True:
        j -= 1
        if row[j] > prevj:
            break
        prevj = row[j]
    y1,y2,y3,x1,x2,x3 = prevj,height,previ,j,indextop,i
    a = ((y3-y2)/(x3-x2)-(y3-y1)/(x3-x1))/((x3**2-x2**2)/(x3-x2)-(x3**2-x1**2)/(x3-x1))
    b = (y3-y2-a*(x3**2-x2**2))/(x3-x2)
    c = y1-a*x1**2-b*x1
    return np.sqrt(b**2-4*a*(c-y2/2))/a
        
    
def findfwhmonrow(newrow, topslist, row, startstreamer):
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
                newrow[indextop] += quadraticfwhmextrapolation(indextop,row) 
        except:
            newrow[indextop] += 0

        
def findtops(newrow, startstreamer, stopstreamer, row, maxvalbackground):
    localmax = maxvalbackground
    localmaxindex = 0
    topslist = []
    printed = False
    for i,item in enumerate(row[startstreamer:stopstreamer]):
        if item > localmax:
            localmax = item
            localmaxindex = i
            printed = False
        elif item < localmax - maxvalbackground:
            localmax = item + 1
            if not printed:
                topslist.append(localmaxindex + startstreamer)
                printed=True
    findfwhmonrow(newrow,topslist,row, startstreamer)    
    
def findfwhm(image, step, twig, maxvalbackground):
    ys = [el[1] for el in twig]
    xs = [el[0] for el in twig]
    fwhms = []
    for i in range(len(xs)-step):
        start = i
        stop = i + step
        y1 = ys[start]
        y2 = ys[stop]
        x1 = xs[start]
        x2 = xs[stop]
        ystep = (x2-x1)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
        xstep = (y1-y2)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
        xm = (x2+x1)/2
        ym = (y2+y1)/2
        points = []
        for j in range(-100,100):
            points.append(getvalue(xm+j*xstep,ym+j*ystep,image))
        topslist = np.zeros(len(points))
        findtops(topslist, 0, -1, points,maxvalbackground)
        fwhms.append(max(topslist[len(points)//2-10:len(points)//2+10]))
    return np.average(fwhms)

if __name__ == '__main__':
    name = 'm/grondmetingen5-73x500ns2025-05-27_14-25-10/grondmetingen5-73x500ns2025-05-27_14-25-10'
    
    with open("settings.yaml") as settings:settings = yaml.safe_load(settings)
    for key in settings.keys():
        if type(settings[key]) == str: settings[key] = eval(settings[key])

    maxdist = settings['MaxDistanceMainBranch']
    steplength = settings['Steplength']
    blurring = settings['BlurrSize']
    maxvalbackground = settings['MaxvalNoise']
    reader = tifffile.imread(f'{name}.ome.tif')
    with open(f'{name}.txt', 'r') as file: branches = [eval(i[:-1]) for i in file.readlines()]
    fwhms = []
    for image,branch in zip(reader,branches):
        rawimage = cv2.GaussianBlur(image,(blurring,blurring),0)
        branchfwhms = []
        for twig in branch:
            startid = findstart(twig, maxdist, True)
            endid = -findstart(twig, maxdist, False)
            if np.isnan(startid) or np.isnan(endid): # If the tops are too far apart, a FWHM is nonsensical
                branchfwhms.append(nan)
                continue
            usedtwig = twig[startid:endid]
            step = steplength
            if len(usedtwig) < step:
                step = len(usedtwig)
            branchfwhms.append(findfwhm(rawimage,step,twig,maxvalbackground))
        fwhms.append(branchfwhms)
    with open(f'{name}_fwhm.txt','w') as txtfile:
        for el in fwhms:
            txtfile.write(str(el)+'\n')