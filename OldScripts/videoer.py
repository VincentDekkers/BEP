import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile


reader = tifffile.imread('Metingen2025-05-22/vaagverschijnsel2025-05-22/vaagverschijnsel2025-05-22.ome.tif')
maxx = np.max(reader)
_,w,h = reader.shape
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (h,w))

cNorm = colors.Normalize(vmin=0, vmax=maxx)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='nipy_spectral')
for values in reader:
    values = scalarMap.to_rgba(values)[:,:,:3]
    values *= 255
    values = values.astype(np.uint8)
    for _ in range(8):
        out.write(values) # frame is a numpy.ndarray with shape (1280, 720, 3)
out.release()
