from numpy import *
import numpy as np
from numpy.random import *
from numpy.linalg import *
from sklearn.cluster import KMeans
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

def find_k_representatives(darray,k):
    kmeans = KMeans(n_clusters=k).fit(darray)
    means = array(kmeans.cluster_centers_)
    #print "Means\n",means.shape
    print "Means\n",means
    dist = darray - means[0]
    dist = dist.dot(dist.T)
    argsort_dist = argsort(diag(dist))
    median_centers = darray[argsort_dist][0]
    #print median_centers
    for i in range(1,k):
        dist = darray - means[i]
        dist = dist.dot(dist.T)
        print diag(dist)
        argsort_dist = argsort(diag(dist))
        median = darray[argsort_dist][0]
        #print median
        median_centers = concatenate((median_centers,median),axis=0)
        #print median_centers
    print "returning"
    return median_centers.reshape(k,2)


dim = 2 # point dimensions
datapoints = 3010
X = randint(1000,size=(datapoints,dim)) # Data
m = 30 # chunk size and memory size limit
i_chunks = int(ceil(float(datapoints)/m)) # i chunks
#print i

k = 6 # k representives
levels = int(log(datapoints)/log(k))
print "Levels:",levels
init_levels = [False]*levels # check if any level was previously initialized or not

memory = dict()

Di = X[0:1*m]
memory[0] = find_k_representatives(Di,k)
init_levels[0] = True

plt.ion()
for j in range(1,i_chunks):
    if j<i_chunks-1:
        Di = X[j*m:(j+1)*m]
    else:
        Di = X[(i_chunks-1)*m:]
    k_reps = find_k_representatives(Di,k)
    #print init_levels[0]
    if init_levels[0] is False:
        memory[0] = k_reps
        init_levels[0] = True
    else:
        #print "BEFORE APPENDING:",memory[0].shape,memory[0]
        memory[0] = append(memory[0],k_reps,axis=0)
        #print "AFTER APPENDING:",memory[0].shape,memory[0]
    for i in range(levels-1):
        m_size = memory[i].shape[0]
        if m_size + k > m:
            if init_levels[i+1] is False:
                memory[i+1] = find_k_representatives(memory[i],k)
                init_levels[i+1] = True
                memory[i] = None
                init_levels[i] = False
                print "Updating level:",i+1
            else:
                print "Updating and adding level",i+1
                memory[i+1] = append(memory[i+1],find_k_representatives(memory[i],k),axis=0)
                memory[i] = None
                init_levels[i] = False
        else:
            break
    plt.clf()
    if j<i_chunks-1:
        plt.plot(X[0:(j+1)*m,0],X[0:(j+1)*m,1],'yo')
    else:
        plt.plot(X[:,0],X[:,1],'yo')
    print init_levels
    for h,i in enumerate(reversed(init_levels)):
        if i is True:
            print h,j
            asas = levels - h - 1
            plt.plot(memory[asas][:,0],memory[asas][:,1],'ks',ms=7)
            break
    #print memory
    plt.pause(0.05)
print "LOOP BREAKED"
print init_levels
print memory[levels-2].shape
final_cluster = memory[0]
for i in range(1,levels-1):
    if memory[i] is not None:
        final_cluster = append(final_cluster,memory[i],axis=0)
print final_cluster.shape
k_reps = find_k_representatives(final_cluster,k)
print k_reps.shape
plt.clf()
plt.plot(X[:,0],X[:,1],'yo')
plt.plot(k_reps[:,0],k_reps[:,1],'ks',ms=7)
plt.pause(200)
