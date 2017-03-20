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

percent = 5
total = 256*256
gray = color.rgb2gray(mpimg.imread("test_image.jpg"))
r = np.random.randint(256,size=(total*percent/100,2))
r1,c1 = r.shape

for i in range(0,r1):
    gray[r[i,0],r[i,1]] = np.nan
Lambda = 0.0625 # close to the default one, but works better
tic = time.time()

# plt.imshow(gray, cmap = plt.cm.Greys_r)
# plt.show()

rpca = R_pca(gray)
L, S = rpca.fit(max_iter=10000, iter_print=100)
toc = time.time()

plt.imshow(L, cmap = plt.cm.Greys_r)
plt.show()
plt.imshow(S, cmap = plt.cm.Greys_r)
plt.show()
plt.imshow(L+S, cmap = plt.cm.Greys_r)
plt.show()

gray = color.rgb2gray(mpimg.imread("test_image.jpg"))
rmse = sqrt(mean_squared_error(L,gray))
print "rmse",rmse
print "time",toc-tic
#rpca.plot_fit()
#plt.show()
