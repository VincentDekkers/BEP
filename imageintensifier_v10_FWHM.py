import imageio.v2 as iio
import scipy.signal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
from multiprocessing import Process

def plotline(coords, title):
    plt.plot(coords)
    plt.title(title)
    plt.show()

def showheatmap(image):
    ax = sns.heatmap(image, cmap='nipy_spectral')
    ax.invert_yaxis()
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
   
def getvalue(x,y, img):
    try:
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        v1,v2,v3 = img[y1][x1],img[y1][x1+1],img[y1+1][x1]
        return v1 + (v2-v1)*(x%1)+(v3-v1)*(y%1)    
    except:
        return 0

def findfwhmonbranches(
    file = "vincenttest_2404_750_10_1000ns_200mbar_firstpulse5us_008.tif",
    rigidness = 0.98,
    displacementfactor = 1.,
    minval = 7.,
    imagefilterblurwindow = 3,
    blurimg = True,
    firstnlineswithoutstreamer = 50,
    firstntops = 50,
    maxdistmainbrach = 100,
    minimumbranchlength = 30,
    maxdistsubbranch = 200,
    averagewidthbranch = 15,
    maxdistancetostartbranch = 5000,
    printnumberofbranchesfound = True):
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
    
    
    # noise statistics
    img = iio.imread(file)
    imgcopy = img.copy()[:firstnlineswithoutstreamer].flatten()
    mean = np.mean(imgcopy)
    std = np.std(imgcopy)
    
    # filtering the noise from the image
    newimg = np.array([[(el-mean)/std if (el-mean)/std > 0 else 0 for el in row] for row in img])
    if blurimg:
        newimg = cv2.GaussianBlur(newimg,(imagefilterblurwindow,imagefilterblurwindow),0)
    newerimg = np.array([[el if (newimg[i-1][j]+newimg[i-1][j-1]+newimg[i-1][j+1]+newimg[i][j-1]+newimg[i][j+1]+newimg[i+1][j]+newimg[i+1][j+1]+newimg[i+1][j-1])-max(newimg[i-1][j],newimg[i-1][j-1],newimg[i-1][j+1],newimg[i][j-1],newimg[i][j+1],newimg[i+1][j],newimg[i+1][j+1],newimg[i+1][j-1]) > minval else 0 for j,el in enumerate(row[1:-1])] for i,row in enumerate(newimg[1:-1])])
    newestimg = np.array([findstreamers(row) for row in newerimg])
    newestimg2 = np.array([findstreamers(row, hor=False) for row in newerimg.transpose()]).transpose()
    betterimg = np.maximum(newestimg, newestimg2)
    
    # generating a list of coordinates of the tops op the streamer
    coordsoftops = np.array([[j,i] for i,row in enumerate(betterimg) for j, el in enumerate(row) if el > 3])
    finalbranches = []
    firsttops = dict()
    
    # finding the start op the streamer
    for i in range(firstntops):
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
    
    # finding the main branch
    angle = 0
    spinalcoords2 = []
    for item in coordsoftops:
        if abs(item[0]-streamerstart) < 1 and len(spinalcoords2) == 0:
            spinalcoords2.append(item)
        elif len(spinalcoords2) > 0:
            if (item[0]-spinalcoords2[-1][0]-angle*(item[1]-spinalcoords2[-1][1]))**2+(item[1]-spinalcoords2[-1][1])**2 < maxdistmainbrach:
                angle = rigidness*angle + (1-rigidness)*(item[0]-spinalcoords2[-1][0])
                spinalcoords2.append([(displacementfactor*(spinalcoords2[-1][0]+angle*(item[1]-spinalcoords2[-1][1]))+item[0])/(1+displacementfactor),item[1]])
                item[0] = 0
    finalbranches.append(spinalcoords2)
    
    # finding the sub-branches iteratively
    for spine in finalbranches:
        branches = []
        pointsreached = []
        for i,branchpoint in enumerate(spine):
            maxdistance = 10**10
            for point in coordsoftops:
                if point[1] > branchpoint[1]:
                    if distancesquared(point, branchpoint) < maxdistance:
                        maxdistance = distancesquared(point, branchpoint)
                        bestpoint = point
            if maxdistance > maxdistancetostartbranch:
                continue
            angle = (bestpoint[0]-branchpoint[0])/(bestpoint[1]-branchpoint[1])
            spinalcoords = []
            itemlist = []
            for item in coordsoftops:
                if distancesquared(item,bestpoint) < 1 and len(spinalcoords) == 0:
                    spinalcoords.append(bestpoint)
                elif len(spinalcoords) > 0:
                    if (item[0]-spinalcoords[-1][0]-angle*(item[1]-spinalcoords[-1][1]))**2 < maxdistsubbranch and abs(item[0]-spinalcoords[-1][0]) < averagewidthbranch:
                        angle = rigidness*angle + (1-rigidness)*(item[0]-spinalcoords[-1][0])
                        spinalcoords.append([(displacementfactor*(spinalcoords[-1][0]+angle*(item[1]-spinalcoords[-1][1]))+item[0])/(1+displacementfactor),item[1]])
                        itemlist.append(tuple(item))
            if len(spinalcoords) > minimumbranchlength:
                if checkbranch(spinalcoords,betterimg):
                    branches.append(itemlist)
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
                if som > maximumscore:
                    maximumscore = som
                    mainbranch = el
            mainbranches.append(mainbranch)
        for el in mainbranches:
            if len(branches) != 0:
                finalbranches.append(branches[el])
                for item in coordsoftops:
                    if tuple(item) in pointsreached[el]:
                        item[0] = 0
        if printnumberofbranchesfound:
            print('\x1b[2K', end='\r')
            print(f"The number of branches found is {len(finalbranches)}", end='\r')
    if printnumberofbranchesfound:
        print()
    finalbranches = [[tuple(el) for el in branch] for branch in finalbranches]
        
    # averaging duplicate y-values
    for branch in finalbranches:
        toremove = []
        toadd = []
        lasty = 0
        totinthisy = 0
        sumofthisy = 0
        for i, el in enumerate(branch):
            if el[1] == lasty:
                if totinthisy == 1:
                    toremove.append(branch[i-1])
                totinthisy += 1
                sumofthisy += el[0]
                toremove.append(el)
            elif totinthisy > 1:
                toadd.append(tuple([sumofthisy/totinthisy,lasty]))
                lasty = el[1]
                totinthisy = 1
                sumofthisy = el[0]
            else:
                lasty = el[1]
                totinthisy = 1
                sumofthisy = el[0]
        for el in toremove:
            branch.remove(el)
        branch += toadd
    finalbranches = [sorted(branch, key=lambda x: x[1]) for branch in finalbranches]

    # For cool plot of fits
    xstoplot = []
    ystoplot = []
    ctoplot = []
    for branch in finalbranches:
        ys = [el[0] for el in branch[:-1]]
        xs = [el[1] for el in branch[:-1]]
        ys = scipy.signal.savgol_filter(ys,min(51,len(ys)),3)
        colors = []
        for i in range(len(xs)):
            start = i - 1 if i != 0 else 0
            stop = i + 1 if i != len(xs) - 1 else len(xs) - 1
            y1 = xs[start]
            y2 = xs[stop]
            x1 = ys[start]
            x2 = ys[stop]
            ystep = (x2-x1)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
            xstep = (y1-y2)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
            xm = (x2+x1)/2
            ym = (y2+y1)/2
            points = []
            for j in range(-100,100):
                points.append(getvalue(xm+j*xstep,ym+j*ystep,newerimg))
            topslist = np.zeros(len(points))
            findtops(topslist, 0, -1, points)
            colors.append(max(topslist[len(points)//2-10:len(points)//2+10]))

        ctoplot += colors
        xstoplot += xs
        ystoplot += list(ys)
            
    plt.scatter(ystoplot,xstoplot, c=ctoplot, cmap='nipy_spectral')
        
    plt.colorbar() 
    # To show number of branches per y-value
    wherearethebranches = np.array([0]*finalbranches[0][-2][1])
    for branch in finalbranches:
        while len(wherearethebranches)<branch[-2][1]:
            wherearethebranches = np.append(wherearethebranches,0)
        wherearethebranches[branch[0][1]:branch[-2][1]] += 1
    o = Process(target=plotline, args=(wherearethebranches,"Branch density",))
    o.start()
    
    # To show maps of stages in between
    p = Process(target=showheatmap, args=(betterimg,))
    p.start()
    q = Process(target=showheatmap, args=(newerimg,))
    q.start()
    r = Process(target=showheatmap, args=(img,))
    r.start()
    
    plt.show()
    


if __name__ == '__main__':
    findfwhmonbranches()