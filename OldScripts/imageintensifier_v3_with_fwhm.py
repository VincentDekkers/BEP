import imageio.v2 as iio
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from multiprocessing import Process

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
    
def findstreamers(row):
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
                    findtops(newrow, startstreamer, stopstreamer,row)
    else:
        if instreamer:
            newrow[startstreamer:] += 1
            for el in np.where(row[startstreamer:]==np.max(row[startstreamer:]))[0]:
                newrow[el+startstreamer]+=1
    return newrow

if __name__ == '__main__':
    # img = iio.imread("vincenttest_2404_1000_20_100ns_100co2_024.tif")
    img = iio.imread("vincenttest_2404_750_10_1000ns_200mbar_firstpulse5us_008.tif")

    imgcopy = img.copy()[:100].flatten()
    mean = np.mean(imgcopy)
    std = np.std(imgcopy)
    newimg = np.array([[(el-mean)/std if (el-mean)/std > 0 else 0 for el in row] for row in img])
    newimg = cv2.GaussianBlur(newimg,(3,3),0)
    newerimg = np.array([[el if (newimg[i-1][j]+newimg[i-1][j-1]+newimg[i-1][j+1]+newimg[i][j-1]+newimg[i][j+1]+newimg[i+1][j]+newimg[i+1][j+1]+newimg[i+1][j-1])-max(newimg[i-1][j],newimg[i-1][j-1],newimg[i-1][j+1],newimg[i][j-1],newimg[i][j+1],newimg[i+1][j],newimg[i+1][j+1],newimg[i+1][j-1]) > 7 else 0 for j,el in enumerate(row[1:-1])] for i,row in enumerate(newimg[1:-1])])
    newestimg = np.array([findstreamers(row) for row in newerimg])
    # print(np.max(newimg[:100]))
    # print(np.average(newimg[:100]))
    # p = Process(target=showheatmap,args=(newimg,))
    # p.start()
    q = Process(target=showheatmap,args=(newerimg,))
    q.start()
    r = Process(target=showheatmap,args=(newestimg,))
    r.start()
    # print(mean,std)

    
    plt.plot(newerimg[387])
    # plt.hist(np.array(newerimg[:100]).flatten(), bins=20)
    # plt.ylim([0,1000])
    plt.show()
    


