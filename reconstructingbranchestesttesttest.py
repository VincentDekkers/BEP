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
        "Metingen2025-05-27/*/*.txt") if "RecSettings" not in i][q*20:20*q+10]
    brancheses = []
    for name in names:
        # print(name)
        with open(name, 'r') as file:
            brancheses.append([eval(i[:-1]) if i != 'error\n' else [] for i in file.readlines()])
    sidewayss = []
    totbranches = []
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
                totbranches.append(len(finalbranches))
                for branch in finalbranches:
                    # if branch[-1][1]-13 > y:
                    #     y = branch[-1][1]-13
                    
                    dist = np.sqrt(distancesquared(branch[-1], (298, 13)))
                    if dist > maxdist:
                        maxdist = dist
                        
                        y = max([i[1] for i in branch])-13
                        totfrac = (maxdist)
                sidewayss.append(totfrac)
    print(q,np.mean(sidewayss)*11/473,np.mean(totbranches))
    
    