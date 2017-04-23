import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
from skimage import color
import matplotlib
from numpy import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import PPCA

mnist = input_data.read_data_sets("data/", one_hot = True)
x_train = mnist.train.images[:50000]
y_train = mnist.train.labels[:50000]

# only index 1 label
index_1 = y_train[:,1]==1
x_train_1 = x_train[index_1]

# Required Number of Principal axis.
k = 8
#print x_train_1[0].shape
[W,sigma_square,Xn,t_mean,M] = PPCA.em_PPCA(x_train_1[0].reshape(28,28),k)
Xn_M = dot(M,Xn)
print Xn_M.shape
Xn_M = Xn_M.reshape(1,Xn_M.shape[0],Xn_M.shape[1])
print W.shape,sigma_square.shape,Xn.shape,t_mean.shape,M.shape

W = W.reshape(1,W.shape[0],W.shape[1])
sigma_square = sigma_square.reshape(1,sigma_square.shape[0],sigma_square.shape[1])
Xn = Xn.reshape(1,Xn.shape[0],Xn.shape[1])
t_mean = t_mean.reshape(1,t_mean.shape[0],t_mean.shape[1])
M = M.reshape(1,M.shape[0],M.shape[1])

for i in range(1,x_train_1.shape[0]):
	try:
		[W1,sigma_square1,Xn1,t_mean1,M1] = PPCA.em_PPCA(x_train_1[i].reshape(28,28),k)
		Xn_M1 = dot(M1,Xn1)
		Xn_M1 = Xn_M1.reshape(1,Xn_M1.shape[0],Xn_M1.shape[1])
		print i
		if W1.shape[1] == k:
			W1 = W1.reshape(1,W1.shape[0],W1.shape[1])
			sigma_square1 = sigma_square1.reshape(1,sigma_square1.shape[0],sigma_square1.shape[1])
			Xn1 = Xn1.reshape(1,Xn1.shape[0],Xn1.shape[1])
			t_mean1 = t_mean1.reshape(1,t_mean1.shape[0],t_mean1.shape[1])
			M1 = M1.reshape(1,M1.shape[0],M1.shape[1])

			W = concatenate((W,W1))
			sigma_square = concatenate((sigma_square,sigma_square1))
			Xn = concatenate((Xn,Xn1))
			t_mean = concatenate((t_mean,t_mean1))
			M = concatenate((M,M1))
			Xn_M = concatenate((Xn_M,Xn_M1))

	except numpy.linalg.linalg.LinAlgError:
		print "Numpy Error"

np.save('ppca_model_W.npy', W)
np.save('ppca_model_sigma_square.npy', sigma_square)
np.save('ppca_model_Xn.npy', Xn)
np.save('ppca_model_t_mean.npy', t_mean)
np.save('ppca_model_M.npy', M)
np.save('ppca_model_Xn_M.npy', Xn_M)
