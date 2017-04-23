import numpy
import numpy as np
import matplotlib.pyplot as plt
from ppca import PPCA
import matplotlib.image as mpimg
import skimage
from skimage import color
import cv2
from pandas import read_csv
import matplotlib
from numpy import *
#set3 = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors

'''
RGB = cv2.imread("CameraMan.png")

RGB_noise = skimage.util.random_noise(RGB, mode='s&p', seed=None, clip=True, amount=1.0)
#RGB_Gray = color.rgb2gray(RGB_noise)

cv2.imshow('image',RGB_noise)
k = cv2.waitKey(0)
'''

seed = 7
np.random.seed(seed)
'''
def ccd(batch_size=256):

	dataframe = read_csv("fer2013.csv", header=None)
	dataset = dataframe.values
	X  = dataset[1:,1]
	E = []
	E_y = []

	for i in range(len(X)):
		Y = np.fromstring(X[i], dtype=int, sep = ' ')
		Y = np.reshape(Y,(48, 48))
		E.append(Y)

	X_inp = np.array(E)
	#X_train = X_inp.reshape(-1,X_inp.shape[1], X_inp.shape[2],1)
	X_train = X_inp.astype('float32')
	size =  X_train.shape[0]
	i = 0
	batch = batch_size
	while(i<(size-batch_size)):
		#inp_img =   X_train[i:i+batch_size]
		inp_img =   X_train[i:i+batch_size]
		ppca = PPCA(inp_img)
		ppca.fit(d=20, verbose=False)
		component_mat = ppca.transform()
		print component_mat.shape
		i += batch_size
		E_y.append(component_mat)
	X_ppca = np.array(E_y)
	print X_ppca.shape

	inp_img =   X_train[i:]
	ppca = PPCA(inp_img)
	ppca.fit(d=20, verbose=False)
	component_mat = ppca.transform()
	print component_mat.shape
	i += batch_size
ccd()
'''

dataframe = read_csv("fer2013.csv", header=None)
dataset = dataframe.values
X = dataset[1:,1]

E = []


for i in range(len(X)):
	Y = np.fromstring(X[i], dtype=int, sep = ' ')
	Y = np.reshape(Y,(48, 48))
	E.append(Y)

X_inp = np.array(E)
#X_train = X_inp.reshape(-1,X_inp.shape[1], X_inp.shape[2],1)
X_train = X_inp.astype('float32')
print X_inp

inp_img = X_train[0,:,:]
ppca = PPCA(inp_img)
ppca.fit(d=20, verbose=False)
component_mat = ppca.transform()
E_y = component_mat.reshape(1,component_mat.shape[0],component_mat.shape[1])

for i in range(1,len(X_train)):
	print i
	inp_img =   X_train[i,:,:]
	ppca = PPCA(inp_img)
	try:
		ppca.fit(d=20, verbose=False)
		component_mat = ppca.transform()
		shape = component_mat.shape
		component_mat = component_mat.reshape(1,component_mat.shape[0],component_mat.shape[1])
		if shape[1] == 20:
			E_y = concatenate((E_y,component_mat))
	except numpy.linalg.linalg.LinAlgError:
		print "Numpy Error"
X_ppca = np.array(E_y)
print X_ppca.shape
np.save('ppca_model_1000.npy', X_ppca)

'''
plt.figure(figsize=(2,2))
plt.imshow(inp_img, cmap=matplotlib.cm.gray)
plt.show()
'''


'''
X = np.random.randint(1,51,size=(500,500))
noise_input = np.random.uniform(-1.0, 1.0, size=[500, 100])
X = X + noise_input

fa = FactorAnalysis()
'''
