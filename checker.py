import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import cv2
import tifffile
import seaborn as sns

name = "Metingen2025-06-10/fullrange67mbar6-00kV500ns2025-06-10_15-34-42/fullrange67mbar6-00kV500ns2025-06-10_15-34-42"
reader = tifffile.imread(f'{name}.ome.tif')
with open(f'{name}.txt', 'r') as file: data = [eval(i[:-1]) for i in file.readlines()]
# reader = tifffile.imread('metignen13-05-2025/testvncent67mbarVO2025-05-13_15-34-35/testvncent67mbarVO2025-05-13_15-34-35.ome.tif')
for i,image in enumerate(reader[9:10],9):
    # imgtoplot = np.minimum(50,image)
    # sns.heatmap(imgtoplot,cmap='nipy_spectral')
    # figManager = plt.get_current_fig_manager()
    # figManager.full_screen_toggle()
    # plt.show()
    xstoplot = []
    ystoplot = []
    ctoplot = []
    for j,branch in enumerate(data[i]):
        xs = [el[0] for el in branch[:-1]]
        ys = [el[1] for el in branch[:-1]]
        
        # rs = [np.sqrt(distancesquared(el, (298,13))) for el in branch[:-1]]
        # phis = [calculateangle((298,13),el) for el in branch[:-1]]
        # rs = scipy.signal.savgol_filter(rs,min(7,len(rs)),1)
        # phis = scipy.signal.savgol_filter(phis,min(7,len(phis)),1 )
        # xs = [r*np.sin(phi)+ 298 for r,phi in zip(rs,phis)][:-1]
        # ys = [r*np.cos(phi)+ 13 for r,phi in zip(rs,phis)][:-1]
        
        colors = [j for _ in xs]
        
        # colors = []
        # for i in range(len(xs)):
        #     start = i - 1 if i != 0 else 0
        #     stop = i + 1 if i != len(xs) - 1 else len(xs) - 1
        #     y1 = ys[start]
        #     y2 = ys[stop]
        #     x1 = xs[start]
        #     x2 = xs[stop]
        #     ystep = (x2-x1)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
        #     xstep = (y1-y2)*np.sqrt(1/(((y2-y1)**2)+((x2-x1)**2)))
        #     xm = (x2+x1)/2
        #     ym = (y2+y1)/2
        #     points = []
        #     for j in range(-100,100):
        #         points.append(getvalue(xm+j*xstep,ym+j*ystep,rawerimage))
        #     topslist = np.zeros(len(points))
        #     findtops(topslist, 0, -1, points)
        #     colors.append(np.log(max(topslist[len(points)//2-10:len(points)//2+10])))
        # for el in rs:
        #     try:
        #         colors.append(wherearethebranches[int(el)])
        #     except:
        #         colors.append(1)
        ctoplot += colors
        xstoplot += list(xs)
        ystoplot += list(ys)
    fig,axs = plt.subplots(1,2)
    ax = sns.heatmap(image,cmap='nipy_spectral',ax=axs[0], cbar=False)
    axs[1].scatter(xstoplot, ystoplot, c=ctoplot, cmap='nipy_spectral')
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    plt.xlim([0,700])
    plt.ylim([0,550])
    axs[1].invert_yaxis()
    axs[0].axis('off')
    axs[1].axis('off')
    # axs[0].ylabel("Distance (pixels)")
    # axs[0].xlabel("Distance (pixels)")
    # axs[1].ylabel("Distance (pixels)")
    # axs[1].xlabel("Distance (pixels)")
    plt.show()