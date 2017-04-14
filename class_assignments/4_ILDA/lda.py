import numpy as np
from numpy import *
from numpy.random import *
from numpy.linalg import *
from numpy.matlib import *
"""
DISCRIMINANT EIGENSPACE MODEL (DEM)
c = classification of respective X rows
omega = (Sw, Sb, X_bar,   N)
M classes
(variables used in code)
Y = classification of respective X rows
omega = (Sw, Sb, X_bar, row)
"""
val_limit = 50
row = 200 # N training samples
col = 50 # Dimensions
M = 3 # M classes (0,1,2)
X = randint(val_limit,size=(row,col))
Y = randint(M,size=row) # classes of training data
# print X
# print Y
Xc_bar = zeros((M,col))
Sw = zeros((col,col))
Sb = zeros((col,col))
X_bar = mean(X,axis=0)
nc = []
for i in range(M):
    # print Y==i
    Xc_bar[i] = mean(X[Y==i],axis=0)
print "MEAN VECTORS:\n", Xc_bar

for i in range(0, M):
    tmp = X[Y==i] - Xc_bar[i]
    Sw += tmp.T.dot(tmp) # shortcut over the iterative row wise append and addition,
    #                                refer to step 2 on the basics at
    #                                http://sebastianraschka.com/Articles/2014_python_lda.html
print "WITHIN MATRIX:\n", Sw
for i in range(0, M):
    Nc = X[Y==i].shape[0]
    nc.append(Nc)
    tmp = Xc_bar[i] - X_bar # Xc_bar = horizontal mean vectors
    Sb += Nc*tmp.T.dot(tmp) # Same shortcut as previously documented comment

print "BETWEEN MATRIX:\n", Sb
D = inv(Sw).dot(Sb)
eigvals, eigvects = eig(D)

U = eigvects[:] # Discriminant Eigenvectors

"""
SEQUENTIAL INCREMENTAL LINEAR DISCRIMINANT ANALYSIS
Let y be a new (N + 1)th training sample with class label k.
Using only omega and y, get updated DEM
omega' = (Sw', Sb', X_bar', N+1) using only omega and y
"""
y = randint(val_limit, size=(1, col))
y_class = randint(M + 1, size=1)
y_class = y_class[0]
N = row
Xc_bar_new = None
X_bar_new = (N * X_bar + y)/(N + 1)
Sb_new = None
Sw_new = None
if y_class == M: # if the new sample belongs to a completely new class

    tmp = y - X_bar_new # horizontal vector

    Sb_new = Sb + tmp.T.dot(tmp)
    Xc_bar_new = concatenate((Xc_bar,y),axis=0)
    Sw_new = Sw[:]

    print "NEW WITHIN MATRIX:\n", Sw_new
    print "NEW BETWEEN MATRIX:\n", Sb_new
    print tmp.T.dot(tmp).shape
    print "\nNEW TRAINING SAMPLE BELONGS TO NEW CLASS\n"
    """
    The Within matrix (Sw) wouldn't change because when subtracting the data matrix of the new class with the mean of the new class,
    the resultant that needs to be added to the Sw would be a zero matrix.
    """
else: # if the new sample belongs to an already existing class
    Xc_bar_new = Xc_bar[:]
    Xc_bar_new[y_class] = 1.0/(nc[y_class]+1)*(nc[y_class]*Xc_bar[y_class] + y)
    nc[y_class] += 1
    Sb_new = zeros((col,col))
    for i in range(0, M):
        tmp = Xc_bar_new[i] - X_bar_new
        Sb_new += nc[i]*tmp.T.dot(tmp)
    Sw_new = Sw[:]
    tmp = y - Xc_bar_new[y_class]
    Sw_new += (nc[y_class]/(nc[y_class] + 1.0))*tmp.T.dot(tmp)
    print "NEW WITHIN MATRIX:\n", Sw_new
    print "NEW BETWEEN MATRIX:\n", Sb_new

D_new = inv(Sw_new).dot(Sb_new)
eigvals_new, eigvects_new = eig(D_new)
print "Norm Error:",norm(D-D_new)


def compute_lda(row, col, M, X, Y):
    Xc_bar = zeros((M,col))
    Sw = zeros((col,col))
    Sb = zeros((col,col))
    X_bar = mean(X,axis=0)
    nc = []
    for i in range(M):
        # print Y==i
        Xc_bar[i] = mean(X[Y==i],axis=0)
    print "MEAN VECTORS:\n", Xc_bar

    for i in range(0, M):
        tmp = X[Y==i] - Xc_bar[i]
        Sw += tmp.T.dot(tmp) # shortcut over the iterative row wise append and addition,
        #                                refer to step 2 on the basics at
        #                                http://sebastianraschka.com/Articles/2014_python_lda.html
    print "WITHIN MATRIX:\n", Sw
    for i in range(0, M):
        Nc = X[Y==i].shape[0]
        nc.append(Nc)
        tmp = Xc_bar[i] - X_bar # Xc_bar = horizontal mean vectors
        Sb += Nc*tmp.T.dot(tmp) # Same shortcut as previously documented comment

    print "BETWEEN MATRIX:\n", Sb
    D = inv(Sw).dot(Sb)
    eigvals, eigvects = eig(D)

    U = eigvects[:] # Discriminant Eigenvectors
