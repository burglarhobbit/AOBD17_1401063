from numpy import *
import numpy as np
from numpy.random import *
from numpy.linalg import *
from sklearn.cluster import KMeans
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

def find_k_representatives(darray,k):
    dist = []
    for i in darray:
        m_array = darray[:] - i
        dist.append(norm(m_array))
    dist = array(dist)
    argsort_dist = argsort(dist)
    first_k = darray[argsort_dist][:k]
    return first_k

dim = 2 # point dimensions
datapoints = 3010
X = randint(3000,size=(datapoints,dim)) # Data
m = 30 # chunk size and memory size limit
i = int(ceil(float(datapoints)/m)) # i chunks
#print i

k = 6 # k representives
Di = X[0:1*m]
memory = find_k_representatives(Di,k)
plt.ion()

for j in range(0,i):
    if j<i-1:
        Di = X[j*m:(j+1)*m]
    else:
        Di = X[(i-1)*m:]
    k_reps = find_k_representatives(Di,k)
    m_size = memory.shape[0]
    if m_size + k > m:
        memory = find_k_representatives(memory,k)
        print "s",memory.shape
    memory = append(memory,k_reps,axis=0)
    print "a",memory.shape
    plt.clf()
    if j<i-1:
        plt.plot(X[0:(j+1)*m,0],X[0:(j+1)*m,1],'yo')
    else:
        plt.plot(X[:,0],X[:,1],'yo')
    plt.plot(k_reps[:,0],k_reps[:,1],'ks',ms=7)
    plt.pause(100.05)
    break
"""
k_reps = find_k_representatives(memory,k)
plt.clf()
plt.plot(X[:,0],X[:,1],'yo')
plt.plot(k_reps[:,0],k_reps[:,1],'ks',ms=7)
"""
