import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import *
import numpy as np
from r_pca import R_pca
import skimage
from skimage import color
from numpy import *
import time
from sklearn.metrics import *

RGB = mpimg.imread("test_image.jpg")
RGB_gray = color.rgb2gray(RGB)
#print(RGB)
#plt.imshow(RGB)
#plt.show()
#RGB_noise = mpimg.imnoise(RGB,'salt & pepper',0.02)

#print(img)
RGB_noise = skimage.util.random_noise(RGB, mode='s&p', seed=None, clip=True, amount=0.30)
corr_img = color.rgb2gray(RGB_noise)
print corr_img.shape
# apply Robust PCA
print corr_img
Lambda = 0.0625 # close to the default one, but works better
tic = time.time()
#plt.imshow(X, cmap = plt.cm.Greys_r)
#plt.show()

rpca = R_pca(corr_img)
L, S = rpca.fit(max_iter=10000, iter_print=100)
toc = time.time()
plt.imshow(L, cmap = plt.cm.Greys_r)
plt.show()
plt.imshow(S, cmap = plt.cm.Greys_r)
plt.show()
plt.imshow(L+S, cmap = plt.cm.Greys_r)
plt.show()
rmse = sqrt(mean_squared_error(RGB_gray,L))
print "rmse",rmse
print "time",toc-tic

#rpca.plot_fit()
#plt.show()
