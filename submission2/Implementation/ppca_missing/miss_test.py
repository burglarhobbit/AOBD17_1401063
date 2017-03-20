import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import inv
import numpy as np

import skimage
from skimage import color
from ppca_mv import ppca_mv
from math import *
from sklearn.metrics import *
from time import time
percent = 30
total = 256*256
gray = color.rgb2gray(mpimg.imread("test_image.jpg"))
r = np.random.randint(256,size=(total*percent/100,2))
r1,c1 = r.shape

for i in range(0,r1):
    gray[r[i,0],r[i,1]] = np.nan

hidden = np.isnan(gray)
missing = np.count_nonzero(hidden)

tic = time()
[a,b,c,d,e]=ppca_mv(gray,20,1)
toc = time()
plt.imshow(gray, cmap = plt.cm.Greys_r)
plt.show()

plt.imshow(e, cmap = plt.cm.Greys_r)
plt.show()

gray = color.rgb2gray(mpimg.imread("test_image.jpg"))
rmse = sqrt(mean_squared_error(e,gray))
print "rmse",rmse
print "time",toc-tic
