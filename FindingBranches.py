# By Vincent Dekkers
# Read the read.me for more information
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import cv2
from multiprocessing import Process, Manager
import tifffile
import time
import glob
import matplotlib
import json
import yaml
from collections import deque

def rotate45(lijst):
    """
    Docstring for rotate45

    Rotates an n x m matrix by 45 degrees clockwise. Returns inhomogenuous list.
    
    For example: 
    [[1, 2, 3, 4]
     [5, 6, 7, 8]
     [9,10,11,12]]
    
    becomes:
    [[1]
     [5, 2]
     [9, 6, 3]
     [10,7, 4]
     [11,8]
     [12]]    
    
    :param lijst: m x n matrix
    """
    l,hh = len(lijst[0]), len(lijst)
    return [[lijst[i][h-i] for i in range(hh) if 0 <= h - i < l] for h in range(l + hh -1)]

def rotatereverse45(lijst):
    """
    Docstring for rotatereverse45

    Rotates an n x m matrix by 45 degrees anticlockwise. Returns inhomogenuous list.
    
    For example: 
    [[1, 2, 3, 4]
     [5, 6, 7, 8]
     [9,10,11,12]]
    
    becomes:
    [[4]
     [3, 8]
     [2, 7, 12]
     [1, 6, 11]
     [5,10]
     [9]]    
    
    :param lijst: m x n matrix
    """
    l,hh = len(lijst[0]), len(lijst)
    return [[lijst[i][i-h] for i in range(hh) if 0 <= i-h < l] for h in range(-l+1,hh)]

def rotate45reverse(lijst, originalshape):
    """
    Docstring for rotate45reverse
    
    Inverts rotate45.
    
    :param lijst: list in the shape of the output of rotate45
    :param originalshape: tuple with length and height of original matrix
    """
    return np.array([[lijst[h+l][item] for l,item in enumerate(np.minimum(range(originalshape[1]-1,-1,-1),h))] for h in range(originalshape[0])])

def rotatereverse45reverse(lijst, originalshape):
    """
    Docstring for rotatereverse45reverse
    
    Inverts rotatereverse45.
    
    :param lijst: list in the shape of the output of rotatereverse45
    :param originalshape: tuple with length and height of original matrix
    """
    enumerate_reversed = lambda l: zip(range(len(l)-1, -1, -1), reversed(l))
    return np.array([[lijst[h+l][item] for l,item in enumerate_reversed(np.minimum(range(originalshape[1]-1,-1,-1),h))] for h in range(originalshape[0])])


def showheatmap(image):
    """
    Docstring for showheatmap
    
    Shows heatmap for a given image, useful tool for debugging purposes.
    
    :param image: m x n matrix
    """
    plt.imshow(image, cmap='nipy_spectral')
    plt.gca().yaxis.set_inverted(True)
    plt.show()
    
def logger(lijst, starttime, speed, checkmorebranches, alg):
    '''
    Logs the progress of individual processes to the termimal. 
    This function should not be part of the main program but rather be running as a seperate process.
    '''
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
            if item[0] == 0 or item[0] > item[1]:
                numberofstringstoprint += 1
                if alg == 1:
                    stringtoprint += f'{i+1}: Number of branches found: {item[0]}, checking {item[1]}'+'\n'
                elif alg == 2:
                    stringtoprint += f'{i+1}: Busy finding branches'+'\n'
                started += 1
            elif item[0] == item[1] and checkmorebranches and alg == 1:
                numberofstringstoprint += 1
                stringtoprint += f'{i+1}: Number of branches found: {item[0]}, looking for streamers with breaks'+'\n'
                started += 1
            else:
                finished += 1
        firstline = f'Finished {finished} out of {length} pictures' +'\n'
        secondline = f'Seconds busy with this set: {int(time.time()-starttime)}' + '\n'
        print('\033[1A\x1b[2K'*prevstringstoprint+firstline+secondline+stringtoprint)
        time.sleep(speed)
        prevstringstoprint = numberofstringstoprint + 3

        
def findtops(newrow, startstreamer, stopstreamer, row):
    """
    Docstring for findtops
    
    Finds the tops in a given row of the image
    
    :param newrow: returned row of the height profile
    :param startstreamer: index at which the streamer starts
    :param stopstreamer: index at which the streamer ends
    :param row: Original row of image
    """
    localmax = 1
    localmaxindex = 0
    printed = False
    for i,item in enumerate(row[startstreamer:stopstreamer]):
        if item > localmax:
            localmax = item
            localmaxindex = i
            printed = False
        elif item < localmax - 1:
            localmax = item + 1
            if not printed:
                newrow[localmaxindex + startstreamer] += 10
                i = 1
                try:
                    while row[localmaxindex + startstreamer + i] == row[localmaxindex + startstreamer]:
                        newrow[localmaxindex + startstreamer + i] += 10
                        i += 1
                except:
                    pass
                printed=True
    
def findstreamers(row, maxvalnoise):
    """
    Docstring for findstreamers
    
    Finds the indexes where the streamer exists. Calls find tops to find the tops on these segments.
    
    :param row: Original row of the image
    :param maxvalnoise: The highest value of the background noise, the program assumes all values above this value to be part of the streamer.
    """
    newrow = np.zeros(len(row))
    if len(newrow) < 20:
        return newrow
    instreamer = False
    time = 0
    for i,item in enumerate(row):
        if not instreamer:
            if item > maxvalnoise:
                if time == 0:
                    startstreamer = i
                time += 1
                if time > 3:
                    instreamer = True
            elif time != 0:
                time -= 1  
        else:
            if item > maxvalnoise and time < 3:
                time += 1
            else:
                if time == 3:
                    stopstreamer = i
                time -= 1
                if time == 0:
                    instreamer = False
                    newrow[startstreamer:stopstreamer] += 1
                    findtops(newrow, startstreamer, stopstreamer,row)
    else:
        if instreamer:
            newrow[startstreamer:] += 1
    return newrow

def distancesquared(point1, point2):
    """
    Docstring for distancesquared
    
    Calculates the distance squared between two points.
    
    :param point1: Tuple[x,y] with coordinates of a point
    :param point2: Tuple[x,y] with coordinates of a point
    """
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2

def distanceparralel(angle, oldpoint, newpoint):
    """
    Docstring for distanceparralel
    
    Calculates the distance between a point and a line (represented by a direction and point of origin)
    
    :param angle: float between -pi and pi with 0 being straight down
    :param oldpoint: Tuple[x,y] with coordinates of a point from where the line originates
    :param newpoint: Tuple[x,y] with coordinates of a point of which the distance is measured
    """
    return (newpoint[0]-oldpoint[0])*np.sin(angle)+(newpoint[1]-oldpoint[1])*np.cos(angle)

def checkbranch(branch, streamer):
    """
    Docstring for checkbranch
    
    Checks whether the found streamer passes through too much 'low areas'.
    
    :param branch: List[coordinates] for which branch to check this
    :param streamer: m x n matrix on which high and low areas are defined.
    """
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
    """
    Docstring for getvalue
    
    Interpolates sub-pixel values using bi-linear interpolation.
    
    :param x: x value
    :param y: y value
    :param img: m x n matrix with image of the streamer
    """
    try:
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        v1,v2,v3 = img[y1][x1],img[y1][x1+1],img[y1+1][x1]
        return v1 + (v2-v1)*(x%1)+(v3-v1)*(y%1)    
    except:
        return 0
    
def includedarkmeasurement(image, settings):
    """
    Docstring for includedarkmeasurement
    
    Substracts the dark measurement of the original image for quality improvements.
    
    :param image: m x n matrix of the image
    :param settings: Dict with the settings as defined in settings.yaml
    """
    try:
        darkimage = tifffile.imread(settings['DarkMeasurementPath'])[0]
        return substractwithoutunderflow(image,darkimage)
    except FileNotFoundError:
        print("Dark path was not found, continuing without dark measurement."+"\n"*10)
        return image
    except:
        print("Incorrect drak measurement, continuing without dark measurement."+"\n"*10)
        return image
    
def substractwithoutunderflow(image,darkimage):
    """
    Docstring for substractwithoutunderflow
    
    Calculates the difference between two images. 
    Due to the matrices' values being uint16, underflow needed to be accounted for.
    Therefore, all negative values are set to 0. (Since uint16 cannot handle negative numbers)
    
    :param image: m x n matrix of the image
    :param darkimage: m x n matrix of the dark measurement
    """
    return np.array([[j-jdark if j>jdark else 0 for j,jdark in zip(i,idark)] for i,idark in zip(image,darkimage)],dtype=image.dtype)

def calculateangle(startpoint,endpoint):
    """
    Docstring for calculateangle
    
    Calculates the angle of the line from startpoint to endpoint
    
    :param startpoint: Tuple[x,y] with coordinates of a point
    :param endpoint: Tuple[x,y] with coordinates of a point
    """
    return np.arctan2(endpoint[0]-startpoint[0],endpoint[1]-startpoint[1])

def filtercamera2(image, settings, pastblurr = False):
    """
    Docstring for filtercamera2
    
    An additional filter to get rid of more noise, specifically for camera 2.
    This looks at the values of the neighboring cells and determines whether they are significant over the noise.
    
    :param image: m x n matrix of the image
    :param settings: Dict with the settings as defined in settings.yaml
    :param pastblurr: Boolean to indicate whether the image has been transformed with a gaussian filter already.
    """
    if not pastblurr:
        rowswithoutstreamer = settings['StartingPoint'][1]//2
        img = image.copy()[:rowswithoutstreamer].flatten()
        mean = np.mean(img)
        std = np.std(img)
        newerimg = np.array([[(el-mean)/std if el > mean else 0 for el in row] for row in image])
    else:
        newerimg = np.array([[el if (image[i-1][j]+image[i-1][j-1]+image[i-1][j+1]+image[i][j-1]+image[i][j+1]+image[i+1][j]+image[i+1][j+1]+image[i+1][j-1])-max(image[i-1][j],image[i-1][j-1],image[i-1][j+1],image[i][j-1],image[i][j+1],image[i+1][j],image[i+1][j+1],image[i+1][j-1]) > settings['MinvalFilterCamera2'] else 0 for j,el in enumerate(row[1:-1])] for i,row in enumerate(image[1:-1])])
    return newerimg

def findfwhmonbranches(image,
    num,
    values,
    progresslist,
    settings
    ):
    """
    Docstring for findfwhmonbranches
    
    Filters the image to a list of coordinates of the tops and passes this to the relevant algorithm.
    
    :param image: m x n matrix of the image
    :param num: id of the process, for the logger
    :param values: global list for extracting the branches afterwards
    :param progresslist: global list read by the logger to log the progess of all subprocesses
    :param settings: Dict with the settings as defined in settings.yaml
    """
    try:
    
        startingpoint = settings['StartingPoint']
        startingpointx = startingpoint[0]
        startingpointy = startingpoint[1]
        blurrsize = settings['BlurrSize']
        maxvalnoise = settings['MaxvalNoise']
        
        if settings['DarkMeasurement']:
            image = includedarkmeasurement(image, settings)
        
        if settings['Camera'] == 2 and settings['FilterCamera2']:
            image = filtercamera2(image, settings)
        
        image = cv2.GaussianBlur(image,(blurrsize,blurrsize),0) #filters with gaussian blur
        
        if settings['Camera'] == 2 and settings['FilterCamera2']:
            image = filtercamera2(image, settings, pastblurr = True)
        
        if settings['NumberOfDirections'] == 1:
            image = np.array([findstreamers(row, maxvalnoise) for row in image])
            coordsoftops = np.array([[j,i] for i,row in enumerate(image) for j, el in enumerate(row) if el > 10])
            
        elif settings['NumberOfDirections'] == 2:
            image1 = np.array([findstreamers(row, maxvalnoise) for row in image]) #horizonal check
            image2 = np.array([findstreamers(row, maxvalnoise) for row in image.transpose()]).transpose() # vertical check
            image = np.maximum(image1,image2)
            coordsoftops = np.array([[j,i] for i,row in enumerate(image) for j, el in enumerate(row) if el > 10])
            
        elif settings['NumberOfDirections'] == 4:
            shape = image.shape
            image1 = np.array([findstreamers(row, maxvalnoise) for row in image]) #horizonal check:
            image2 = np.array([findstreamers(row, maxvalnoise) for row in image.transpose()]).transpose() # vertical check
            image3 = rotate45reverse([findstreamers(row, maxvalnoise) for row in rotate45(image)],shape)
            image4 = rotatereverse45reverse([findstreamers(row, maxvalnoise) for row in rotatereverse45(image)],shape)
            image = image1 + image2 + image3 + image4
            coordsoftops = np.array([[j,i] for i,row in enumerate(image) for j, el in enumerate(row) if el > 20]) 
        else:
            raise SyntaxError("The NumberOfDirections must be either 1,2 or 4.")
        
        disttocenter = [(startingpointx-el[0])**2+(startingpointy-el[1])**2 for el in coordsoftops]
        coordsoftops = [x for _,x in sorted(zip(disttocenter, coordsoftops), key=lambda pair: pair[0])]
        if settings['Algorithm'] == 1:
            classicalbranchfindingalg(coordsoftops, progresslist, num, settings, values, image)
        elif settings['Algorithm'] == 2:
            newbranchfindingalg(coordsoftops, disttocenter, progresslist, num, settings, values)
        else:
            raise SyntaxError("The Algorithm must be either 1 or 2.")
    except:
        values.append([num,'error'])
        progresslist[num] = [1,1]
        # generating a list of coordinates of the tops op the streamer

def classicalbranchfindingalg(coordsoftops, progresslist, num, settings, values, image):
    """
    Docstring for classicalbranchfindingalg
    
    Original script to connect the tops into a branch structure.
    As explaned in my BEP report.
    
    :param coordsoftops: List with the coordinates of the tops, sorted by distance to the startingpoint.
    :param progresslist: global list read by the logger to log the progess of all subprocesses
    :param num: id of the process, for the logger
    :param settings: Dict with the settings as defined in settings.yaml
    :param values: global list for extracting the branches afterwards
    :param image: m x n matrix of the image
    """
    try:
# Initiates some variables from the settings, for a better performance (lookup is expensive)
        rigidness = float(settings['Rigidness'])
        displacementfactor = settings['DisplacementFactor']
        maxdistmainbrach = settings['MaxDistanceMainBranch']
        minimumbranchlength = settings['MinimumBranchLengh']
        maxdistsubbranch = settings['MaxDistanceSubBranch']
        maxdistancetostartbranch = settings['MaxDistanceToStartBranch']
        correlationcoeff = settings['CorrelationCoeff']
        maxbend = settings['MaxBend']
        maxdistmissedtop = settings['MaxDistmissedTop']
        startingpoint = settings['StartingPoint']
        
# Initiating some variables
        startingpointx = startingpoint[0]
        startingpointy = startingpoint[1]
        finalbranches = []
        angle = 0
        spinalcoords2 = [coordsoftops[0].copy()]
        itemlist = []
        progresslist[num] = [0,0]
        
# Tracks the main branch
        for item in coordsoftops: 
            if np.sqrt(distancesquared(item, spinalcoords2[-1])) - distanceparralel(angle, spinalcoords2[-1], item) < maxdistmainbrach and abs(calculateangle(spinalcoords2[-1],item) - angle) < maxbend and np.sqrt(distancesquared(item,spinalcoords2[-1]))<20:
                angle = rigidness*angle + (1-rigidness)*calculateangle(spinalcoords2[-1],item)
                spinalcoords2.append([(displacementfactor*item[0]+spinalcoords2[-1][0]+distanceparralel(angle,spinalcoords2[-1],item)*np.sin(angle))/(displacementfactor+1),(displacementfactor*item[1]+spinalcoords2[-1][1]+distanceparralel(angle,spinalcoords2[-1],item)*np.cos(angle))/(displacementfactor+1)])
                itemlist.append(tuple(item))
                item[0] = 0
        added = []
        for item in itemlist: # checks intermediate points
            for top in coordsoftops:
                if tuple(top) not in added and distancesquared(top,item) < 10:
                    added.append(tuple(top))
                    top[0] = 0
        itemlist += added
        itemlist = [x for _,x in sorted(zip([(startingpointx-el[0])**2+(startingpointy-el[1])**2 for el in itemlist], itemlist), key=lambda pair: pair[0])]
        finalbranches.append(itemlist)
        
        
        checkedbranches = []
# finding the sub-branches iteratively
        for h,spine in enumerate(finalbranches): 
            branches = []
            pointsreached = []
            
# For each point in a branch ...
            for i,branchpoint in enumerate(spine):
                maxdistance = [10**10]*8
                bestpoints = [0]*8
# ... after finding the closest unused top in each of 8 directions ...
                for point in coordsoftops:
                    quarant = int(calculateangle(branchpoint,point)*4/np.pi+4)
                    if quarant == 8: continue
                    if distancesquared(point, branchpoint) < maxdistance[quarant]:
                        maxdistance[quarant] = distancesquared(point, branchpoint)
                        bestpoints[quarant] = point
# ... a fit is made to find the possible branches.
                for bestpointnumber, bestpoint in enumerate(bestpoints):
                    if maxdistance[bestpointnumber] > maxdistancetostartbranch:
                        continue
                    angle = calculateangle(branchpoint,bestpoint)
                    spinalcoords = []
                    itemlist = []
                    for item in coordsoftops:
                        if distancesquared(item,bestpoint) < 1 and len(spinalcoords) == 0:
                            spinalcoords.append(bestpoint)
                        elif len(spinalcoords) > 0 and item[0] != 0:
                            if np.sqrt(distancesquared(item, spinalcoords[-1])) - distanceparralel(angle, spinalcoords[-1], item) < maxdistsubbranch and abs(calculateangle(spinalcoords[-1],item) - angle) < maxbend and np.sqrt(distancesquared(item,spinalcoords[-1]))<20:
                                angle = rigidness*angle + (1-rigidness)*calculateangle(spinalcoords[-1],item)
                                spinalcoords.append([(displacementfactor*item[0]+spinalcoords[-1][0]+distanceparralel(angle,spinalcoords[-1],item)*np.sin(angle))/(displacementfactor+1),(displacementfactor*item[1]+spinalcoords[-1][1]+distanceparralel(angle,spinalcoords[-1],item)*np.cos(angle))/(displacementfactor+1)])
                                itemlist.append(tuple(item))
                    if len(spinalcoords) > minimumbranchlength:
                        if checkbranch(spinalcoords,image):
                            itemlist.insert(0,tuple(branchpoint))
                            branches.append(itemlist)
                            pointsreached.append(itemlist)
# If no branches are found...
            if len(branches) == 0:
                checkedbranches.append([tuple(element) for element in spine])
                progresslist[num] = [len(finalbranches),h+1]
# ... and all already found branches have already been tested...
                if len(finalbranches) == h+1 and settings['SearchForContinuedStreamers']:
# ... all unused tops are tested away from the starting point as startingangle...
                    for itemm in coordsoftops:
                        if itemm[0] == 0:
                            continue
                        
                        angle = calculateangle(startingpoint,itemm)
                        possiblebranch = []
                        spinalcoords3 = [itemm]
                        for item in coordsoftops: # finds main branch
                            if item[0] != 0:
                                if np.sqrt(distancesquared(item, spinalcoords3[-1])) - distanceparralel(angle, spinalcoords3[-1], item) < maxdistmainbrach and abs(calculateangle(spinalcoords3[-1],item) - angle) < maxbend and np.sqrt(distancesquared(item,spinalcoords3[-1]))<20:
                                    angle = rigidness*angle + (1-rigidness)*calculateangle(spinalcoords3[-1],item)
                                    spinalcoords3.append([(displacementfactor*item[0]+spinalcoords3[-1][0]+distanceparralel(angle,spinalcoords3[-1],item)*np.sin(angle))/(displacementfactor+1),(displacementfactor*item[1]+spinalcoords3[-1][1]+distanceparralel(angle,spinalcoords3[-1],item)*np.cos(angle))/(displacementfactor+1)])
                                    possiblebranch.append(tuple(item))
                        added = []
                        for item in possiblebranch: # checks intermediate points
                            for top in coordsoftops:
                                if tuple(top) not in added and distancesquared(top,item) < 10:
                                    added.append(tuple(top))
                        possiblebranch += added
                        possiblebranch = [x for _,x in sorted(zip([(startingpointx-el[0])**2+(startingpointy-el[1])**2 for el in possiblebranch], possiblebranch), key=lambda pair: pair[0])]
# ... and if one is found, the regular program tries to find more branches in this extention.
                        if len(possiblebranch) > minimumbranchlength:
                            finalbranches.append(possiblebranch)
                            for item in coordsoftops:
                                if tuple(item) in possiblebranch:
                                    item[0] = 0
                            break
                        itemm[0] = 0
                    else:
                        progresslist[num] = [len(finalbranches),h+2]
                continue
# For all branches found branching from a single branch, their correlation is measured.
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
# For all branches found with some kind of overlap, the best fit is calculated.
            mainbranches = []
            indexesofbranching = [0]
            for branch in branchindexes:
                maximumscore = 0
                dist = 10000000
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
                        p1 = branches[el][0:2]
                        dist = (p1[0][0]-p1[1][0])**2+(p1[0][1]-p1[1][1])**2
                    elif som == maximumscore:
                        p1 = branches[el][0:2]
                        dist2 = (p1[0][0]-p1[1][0])**2+(p1[0][1]-p1[1][1])**2
                        if dist > dist2:
                            mainbranch = el
                            dist = dist2
# Looks for the possible missing of some tops.
                added = []
                firstinbranch = branches[mainbranch][0]
                indexesofbranching.append(spine.index(branches[mainbranch][0]))
                for el in branches[mainbranch][1:]:
                    for point in coordsoftops:
                        if tuple(point) not in added and distancesquared(el, point) < maxdistmissedtop:
                            added.append(tuple(point))
                branches[mainbranch] += added
                branches[mainbranch] = list(set(branches[mainbranch]))
                branches[mainbranch] = [firstinbranch] + [x for _,x in sorted(zip([(startingpointx-el[0])**2+(startingpointy-el[1])**2 for el in branches[mainbranch][1:]], branches[mainbranch][1:]), key=lambda pair: pair[0])]
                mainbranches.append(mainbranch)
 
            indexesofbranching.sort()
            for ii,index in enumerate(indexesofbranching):
                try:
                    splitbranch = spine[index:indexesofbranching[ii+1]+1]
                except:
                    splitbranch = spine[index:]
                checkedbranches.append([tuple(el) for el in splitbranch])
# And adds all the used tops to 'used tops', as well as the found branches to unchecked branches, for it to check.
            for el in mainbranches:
                if len(branches) != 0:
                    finalbranches.append(branches[el])
                    for item in coordsoftops:
                        if tuple(item) in pointsreached[el]:
                            item[0] = 0
            progresslist[num] = [len(finalbranches),h+1]
        finalbranches = [[tuple(el) for el in branch] for branch in checkedbranches]
 
        values.append([num,finalbranches])
    except:
        values.append([num,'error'])
        progresslist[num] = [1,1]
    
def makebranchesfromtree(tree, branches, startingpoint, minlengthbranch):
    """
    Docstring for makebranchesfromtree

    Recursive formula for finding a branch structure from a relation dictionary.
    If the startingpoint has no children, a deque (double ended queue (fancy list)) is returned with just that element.
    If the startingpoint has one child, the function is called with the child as a startingpoint and upon return adds itself to the front of the list, and returns that list
    If the startingpoint has more children, a list is made where for each element the function is called with each child as a starting point. The child with the deepest child is used and the rest are saved as sub-branches in the branches list.
    
    :param tree: Relational dictionary parent: [children] with coordinates and their predecessors.
    :param branches: List of all found branches
    :param startingpoint: Startingpoint for the algorithm
    :param minlengthbranch: Parameter from the settings to assert a minimum length for the found branches
    """
    if startingpoint not in tree.keys():
        return deque([startingpoint])
    elif len(tree[startingpoint]) == 1:
        for el in tree[startingpoint]:
            prevbranch = makebranchesfromtree(tree, branches, el, minlengthbranch)
        prevbranch.appendleft(startingpoint)
        return prevbranch
    else:
        prevbranches = []
        for el in tree[startingpoint]:
            prevbranches.append(makebranchesfromtree(tree, branches, el, minlengthbranch))
        lengths = [len(prevbranch) for prevbranch in prevbranches]
        maxlength = max(lengths)
        maxfound = False
        for i,prevbranch in enumerate(prevbranches):
            prevbranch.appendleft(startingpoint)
            if (lengths[i] != maxlength or maxfound) and lengths[i] >= minlengthbranch:
                branches.append(prevbranch)
            elif lengths[i] == maxlength:
                maxfound = True
        else:
            return prevbranches[lengths.index(maxlength)]    

def newbranchfindingalg(coordsoftops, disttocenter, progresslist, num, settings, values):
    """
    Docstring for newbranchfindingalg
    
    Faster algorithm to find a branch structure given some tops.
    For all tops ascending, the closest other top is found that
        - is closer to the startingpoint
        - is less than the maxdistance away 
    (Therefore only tops with a distance between the distance of the original top and that distance - max distance have to be checked)
    
    This closest top is stored in a relational dictionary.
    makebranchesfromtree is then called to make this relational database into a branch structure.
    
    :param coordsoftops: List with the coordinates of the tops, sorted by distance to the startingpoint.
    :param disttocenter: List with distances to the center for all tops.     
    :param progresslist: global list read by the logger to log the progess of all subprocesses
    :param num: id of the process, for the logger
    :param settings: Dict with the settings as defined in settings.yaml
    :param values: global list for extracting the branches afterwards
    """
    progresslist[num] = [1,0]
    disttocenter.sort()
    coordsoftops = [tuple(coord) for coord in coordsoftops]
    closesttops = deque([tuple(np.array(settings['StartingPoint'], dtype=np.int64))])
    closestdistance = deque([0])
    predecessors = {}
    maxdist = settings['MaxDistTopsAlg2']
    for el, dist in zip(coordsoftops, disttocenter):
        bestdist = maxdist
        for point in closesttops:
            if np.sqrt(distancesquared(el,point)) <= bestdist:
                bestdist = np.sqrt(distancesquared(el,point)) 
                bestpoint = point
        if bestdist < maxdist:
            if bestpoint in predecessors.keys():
                predecessors[bestpoint].append(el)
            else:
                predecessors[bestpoint] = [el]
            closesttops.append(el)
            closestdistance.append(dist)
                
        while len(closestdistance)>0 and closestdistance[0] < dist - maxdist: # Triangle inequality
            closestdistance.popleft()
            closesttops.popleft()
    subbranches = []
    mainbranch = makebranchesfromtree(predecessors, subbranches, tuple(np.array(settings['StartingPoint'],dtype=np.int64)), settings['MinimumBranchLengh'])
    subbranches.append(mainbranch)
    subbranches.append(mainbranch)
    subbranches = sorted(subbranches, key=lambda x: -len(x))
    subbranches = [list(el) for el in subbranches]
    subbranches = [[(int(el[0]), int(el[1])) for el in branch] for branch in subbranches]
    values.append([num,subbranches])
    progresslist[num] = [1,1]
    

def finder(file, settings):
    """
    Docstring for finder
    
    Main program for camera 1. 
    This program makes the logger and all the prosesses to multithread everything efficiently.
    
    :param file: path to the .ome.tif file
    :param settings: Dict with the settings as defined in settings.yaml
    """
    reader = tifffile.imread(file) # reads file
    print('\033[1A\x1b[2K', end='\r')
    processes = []
    manager = Manager()
    values = manager.list([])
    progresslist = manager.list([0]*len(reader))
    s = Process(target=logger, args=(progresslist,time.time(),settings['ProgressUpdateTime'],settings['SearchForContinuedStreamers'], settings['Algorithm'])) # initiate logger
    s.start()
    for i,image in enumerate(reader): # for each image
        if np.max(np.max(image)) > 10: # checks for black image
            p = Process(target=findfwhmonbranches, args=(image,i,values,progresslist,settings,)) # initiates processes
            p.start()
            processes.append(p)
        else:
            values.append([i,'error'])
            progresslist[i] = [1,1]
    for p in processes:
        p.join() # waits until termination of all processes
    time.sleep(0.5)
    s.kill()
    return values

def writercamera2(file, num, progresslist, settings,):
    """
    Docstring for writercamera2
    
    Reads the image, sends to finding functions, and writes for camera 1.
    
    :param file: Description
    :param num: Description
    :param progresslist: Description
    :param settings: Description
    """
    image = iio.imread(file)
    values = []
    findfwhmonbranches(image, num, values, progresslist, settings)
    values = [[0,[[(int(coord[0]),int(coord[1])) for coord in branch] for branch in values[0][1]]]]
    
    with open(f'{file[:-4]}.txt','w') as txtfile:
        for el in [j[1] for j in values]:
            txtfile.write(str(el)+'\n')
    with open(f'{file[:-4]}.json','w') as jsonfile:
        json.dump({i:{j:{k: [int(coord) if coord != 'error' else coord for coord in top] for k, top in enumerate(branchno)} for j,branchno in enumerate(picno)} for i,picno in values},jsonfile)
    

def findercamera2(files, settings):
    """
    Docstring for findercamera2
    
    Main program for camera 2. 
    This program makes the logger and all the prosesses to multithread everything efficiently.
    
    :param files: list of paths to the .tif files
    :param settings: Dict with the settings as defined in settings.yaml
    """
    processes = []
    manager = Manager()
    progresslist = manager.list([0]*len(files))
    s = Process(target=logger, args=(progresslist,time.time(),settings['ProgressUpdateTime'],settings['SearchForContinuedStreamers'], settings['Algorithm'])) # initiate logger
    s.start()
    for i,file in enumerate(files):
        # writercamera2(file, i, progresslist, settings,)
        p = Process(target=writercamera2, args=(file, i, progresslist, settings,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    time.sleep(0.5)
    s.kill()

def click_event(event, x, y, flags, params):
    """
    Middleware to save the coordinates of the click when clicking the start of the streamer.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        startingpoint[0] = x
        startingpoint[1] = y
        cv2.destroyAllWindows()
 
# Main funtion of the program   
if __name__ =='__main__':

# Opens the settings
    with open("settings.yaml") as settings:settings = yaml.safe_load(settings)
    for key in settings.keys():
        if type(settings[key]) == str: settings[key] = eval(settings[key])

# Finds the files
    files = glob.glob(f'{settings["Path"]}{".ome" if settings["Camera"] == 1 else ""}.tif') #Searches all files
    startingpoint = settings['StartingPoint']
    if settings["Camera"] == 2:
        files = [el for el in files if '.ome.tif' not in el] # As .ome.tif is caught when looking for .tif
        
# If one wants to click the start of the streamer by hand    
    if settings['Clickstart']:
        if settings['Camera'] == 1:
            reader = tifffile.imread(files[0])
        else:
            reader = [iio.imread(files[0])]
            
        for image in reader:
            if np.max(image) > 10: # Sometimes the first picture is just black due to a camera error
                cmap = matplotlib.colormaps['nipy_spectral']
                image = reader[0]
                image = cmap(image/np.max(image))
                cv2.imshow('streamer', image)
                cv2.setMouseCallback('streamer', click_event)
                cv2.waitKey(0)
                print(f"The chosen top is {startingpoint}")
                break

# Reads and writes for camera 1. Also calls the main script
    if settings['Camera'] == 1:
        for i,file in enumerate(files):
            if len(glob.glob(f'{file[:-8]}.txt')) == 0: # check if braches have been found alr
                print(f'Imageset {i+1} out of {len(files)}',end='\n')
                values = list(finder(file, settings)) # the script
                values.sort()
                with open(f'{file[:-8]}.txt','w') as txtfile:
                    for el in [j[1] for j in values]:
                        txtfile.write(str(el)+'\n')
                with open(f'{file[:-8]}.json','w') as jsonfile:
                    json.dump({i:{j:{k: [int(coord) if coord != 'error' else coord for coord in top] for k, top in enumerate(branchno)} for j,branchno in enumerate(picno)} for i,picno in values},jsonfile)
                print('\033[1A\x1b[2K'*4,end='\r')
# For camera 2, there is only 1 image per file, thus this is faster to let the individual processes write to the disk.
    elif settings['Camera'] == 2:
        findercamera2(files,settings)
    else:
        print("Camera option must either be 1 or 2")
