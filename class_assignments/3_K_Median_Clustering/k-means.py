from sklearn.cluster import KMeans
from numpy import *
import numpy as np
from numpy.random import *
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

X = randint(300,size=(100,2))

plt.ion()
initlabel = None
for i in range(1,1000):
    y = randint(300,size=(1,2))
    X = concatenate((X,y),axis=0)
    kmeans = KMeans(n_clusters=4).fit(X)
    labels = array(kmeans.labels_)
    print labels
    if initlabel is None:
        initlabel = labels[:]
    else:
        dict1 = {}
        for i in range(20):
            dict1[labels[i]] = initlabel[i]
        labels[:initlabel.shape[0]] = initlabel
        labels[-1] = dict1[labels[-1]]
        initlabel = labels[:]
    means = array(kmeans.cluster_centers_)
    """
    cl1 = labels == 0
    cl2 = labels == 1
    cl3 = labels == 2
    cl4 = labels == 3
    cl1 = X[cl1]
    cl2 = X[cl2]
    cl3 = X[cl3]
    cl4 = X[cl4]
    """
    colors = ['red','yellow','blue','magenta']
    plt.clf()
    plt.scatter(X[:,0],X[:,1],c=labels,cmap=mpl.colors.ListedColormap(colors))
    plt.plot(means[:,0],means[:,1],'ks',ms=7)
    plt.pause(0.05)
