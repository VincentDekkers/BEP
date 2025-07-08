import imageio.v2 as iio
import scipy.signal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
from multiprocessing import Process


def showheatmap(image):
    ax = sns.heatmap(image, cmap='nipy_spectral')
    ax.invert_yaxis()
    plt.show()
    
def getvalue(x,y, img):
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    v1,v2,v3 = img[y1][x1],img[y1][x1+1],img[y1+1][x1]
    return v1 + (v2-v1)*(x%1)+(v3-v1)*(y%1)
    

if __name__ == '__main__':
    img = iio.imread("vincenttest_2404_750_10_1000ns_200mbar_firstpulse5us_008.tif")
    imgcopy = img.copy()[:50].flatten()
    mean = np.mean(imgcopy)
    std = np.std(imgcopy)
    newimg = np.array([[(el-mean)/std if (el-mean)/std > 0 else 0 for el in row] for row in img])
    newimg = cv2.GaussianBlur(newimg,(3,3),0)
    newerimg = np.array([[el if (newimg[i-1][j]+newimg[i-1][j-1]+newimg[i-1][j+1]+newimg[i][j-1]+newimg[i][j+1]+newimg[i+1][j]+newimg[i+1][j+1]+newimg[i+1][j-1])-max(newimg[i-1][j],newimg[i-1][j-1],newimg[i-1][j+1],newimg[i][j-1],newimg[i][j+1],newimg[i+1][j],newimg[i+1][j+1],newimg[i+1][j-1]) > 7 else 0 for j,el in enumerate(row[1:-1])] for i,row in enumerate(newimg[1:-1])])
    # showheatmap(newerimg)
    # newerimg = np.array([[(i-500)**2+(j-500)**2-1500 for i in range(1000)] for j in range(1000)])
    # newerimg = np.eye(1000)
    x1 = 630
    y1 = 260
    x2 = 630
    y2 = 270
    xstep = (x2-x1)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
    ystep = (y1-y2)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
    xs = (x2+x1)/2
    ys = (y2+y1)/2
    points = []
    for i in range(-100,100):
        points.append(getvalue(xs+i*xstep,ys+i*ystep,newerimg))
    p = Process(target=showheatmap, args=(newerimg,))
    p.start()
    plt.plot(points)
    plt.show()
        
    