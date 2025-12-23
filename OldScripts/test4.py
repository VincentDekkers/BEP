import cv2
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import tifffile
import numpy as np

src_path = 'm/doublepulse3_N2_50mbar_6kV_2025-07-10_11-41-48/doublepulse3_N2_50mbar_6kV_2025-07-10_11-41-48.ome.tif'
img = tifffile.imread(src_path)[0]
# img = np.minimum(img,1000)
img = cv2.GaussianBlur(img,(9,9),0)
# print(np.max(img))
image = np.gradient(img)
imagex = np.gradient(image[0])
imagey = np.gradient(image[1])
# newimage = imagex[0]+imagex[1]+imagey[0]+imagey[1]
newimage = imagex[0]+imagey[1]
# newimage = imagex[1]+imagey[0]

def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True)
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='nipy_spectral')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()
image = np.sqrt((image[0])**2 + (image[1])**2)

# img = np.log(np.maximum(img,1))
# image = np.log(np.maximum(image,1))
# img = np.minimum(img,500)
# image = np.minimum(image, 100)
# print(img.shape[0])
newimg = [[(newimage[i][j]<1)*img[i][j]**2/image[i][j] if image[i][j] != 0 and img[i][j]>10 else img[i][j] if img[i][j]>10 else 0 for j in range(image.shape[1])] for i in range(image.shape[0])]
# plot_images(img, -newimage*newimg)
# plot_images(img, np.maximum(np.minimum(-newimage*newimg,10),0))
plot_images(img,newimg)
# img = cv2.imread(src_path, 0) # 0 imports a grayscale

# if img is None:
#     raise(ValueError(f"Image didn\'t load. Check that '{src_path}' exists."))

# a, b = detect_ridges(img, sigma=0.0)

# plot_images(img, a, b, a+b)
