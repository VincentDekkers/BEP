import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tifffile
import seaborn as sns

name = "m/grondmetingen5-73x500ns2025-05-27_14-25-10/grondmetingen5-73x500ns2025-05-27_14-25-10"
reader = tifffile.imread(f'{name}.ome.tif')
with open(f'{name}.txt', 'r') as file: data = [eval(i[:-1]) for i in file.readlines()]
for i,image in enumerate(reader):
    xstoplot = []
    ystoplot = []
    ctoplot = []
    for j,branch in enumerate(data[i]):
        xs = [el[0] for el in branch[:-1]]
        ys = [el[1] for el in branch[:-1]]
        colors = [j for _ in xs]
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
    plt.plot(318,79,'ro')
    plt.show()