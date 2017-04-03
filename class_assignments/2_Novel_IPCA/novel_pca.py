"""
Novel PCA by Bhavya Patwa
"""

import numpy as np
from numpy import *
from numpy.linalg import *
from scipy import linalg as sciLA

import numpy
numpy.set_printoptions(threshold=numpy.nan)

row = 20000
col_m = 1000
col_r = 500
X1 = np.random.randint(100,size=(row,col_m))
X2 = np.random.randint(100,size=(row,col_r))
#X1 = np.random.rand(row,col_m)
#X2 = np.random.rand(row,col_r)

X = concatenate((X1,X2),axis=1)

E_X1 = mean(X1,axis=0)
E_X2 = mean(X2,axis=0)
E_X = mean(X,axis=0)
# print E_X1
"""
Compute Sigma2 and Sigma3 according to (2)
"""
Sigma = (X - E_X).transpose().dot(X - E_X)
Sigma1 = (X1 - E_X1).transpose().dot(X1 - E_X1)
Sigma2 = (X1 - E_X1).transpose().dot(X2 - E_X2)
Sigma3 = (X2 - E_X2).transpose().dot(X2 - E_X2)

sig1_eigval,sig1_eigvect = eig(Sigma1)
# print sig1_eigval
positive_order = sig1_eigval>0.0 # Get boolean status of positive eigenvalues
sig1_eigval = sig1_eigval[positive_order] # Lambda
sig1_eigvect = sig1_eigvect[positive_order] # U

"""
Obtain Best rank-k approximation of P_lxm
"""
lambda_tild = sig1_eigval
U_tild = sig1_eigvect

P_lxm = sqrt(diag(lambda_tild)).dot(U_tild.transpose())
l = sig1_eigval.shape[0]

# best rank-k approximation P(1) of P_lxm
k = int(l/100)
k = l
lambda_k = lambda_tild[:k]
V_k = U_tild[:,:k]
I_k = identity(k)
I_k_0 = concatenate((I_k,zeros((l-k,k))),axis=0)
P_1 = I_k_0.dot(sqrt(diag(lambda_k))).dot(V_k.transpose())
# print norm(P_lxm-P_1)
"""
print "l:%d k:%d " %(l,k)
print "Ik:",concatenate((identity(k),zeros((l-k,k))),axis=1).shape
print "lambda:",lambda_k.shape
print "V_k",V_k.transpose().shape
eig whole, Q3=vector*sqrt(eigv)*vectorT
"""

"""
Compute Q1 and Q3 according to (3) and (5)
"""
Q1_lxr = diag(1./sqrt(lambda_tild)).dot(U_tild.transpose()).dot(Sigma2)

# print Sigma2.transpose().shape,U_tild.shape,diag(1./lambda_tild).shape,U_tild.transpose().shape,Sigma2.shape
eigval,eigvect = eig(Sigma3 - Q1_lxr.transpose().dot(Q1_lxr))
eigval = sciLA.sqrtm(diag(eigval))
# print eigval
Q3_rxr = eigvect.dot(eigval).dot(eigvect.transpose())
# Q3_rxr = sciLA.sqrtm(Sigma3 - Q1_lxr.transpose().dot(Q1_lxr))
# Q3_rxr = sqrt(Sigma3 - Sigma2.transpose().dot(U_tild).dot(diag(1./lambda_tild)).dot(U_tild.transpose()).dot(Sigma2))
# print Q3_rxr

"""
tmp = Sigma3 - Q1_lxr.transpose().dot(Q1_lxr)
tmpT_tmp = tmp.transpose().dot(tmp)
tmp_eigval, tmp_eigvect = eig(tmpT_tmp)
Q3_rxr = tmp_eigvect.dot(diag(sqrt(tmp_eigval))).dot(tmp_eigvect.transpose())
"""
Q2 = zeros((Q3_rxr.shape[0],P_1.shape[1]))
upper = concatenate((P_lxm,Q1_lxr),axis=1)
lower = concatenate((Q2, Q3_rxr),axis=1)
asas = concatenate((upper,lower),axis=0)
ghgh = asas.transpose().dot(asas)
#print norm(Sigma-ghgh)

"""
Compute JK as QR decomposition of [ 0_kxk   0 ]               [ Q1 ]
                                  [ 0       I ]               [ Q3 ]
                                              (l + r)x(l + r)
"""
I_l_r = identity(l + col_r)
I_l_r[:k] = 0
Q1_Q3 = concatenate((Q1_lxr,Q3_rxr),axis=0)
J,K = qr(I_l_r.dot(Q1_Q3))

"""
Obtain SVD of smaller matrix [ lambda_k  [ I_k 0 ] [ Q1 ] ]
                                                   [ Q3 ]    = U_hat,lambda_hat,VT_hat
                             [    0               K       ]
"""
tmp1 = concatenate((I_k,zeros((k,l+col_r-k))),axis=1)
tmp2 = tmp1.dot(Q1_Q3) # 2nd quandrant
# print "tmp2",tmp2.shape,"K",K.shape
tmp3 = concatenate((sqrt(diag(lambda_k)),tmp2),axis=1) # upper half of step 5
tmp4 = concatenate((zeros((K.shape[0],k)),K),axis=1) # lower half of step 5
U_hat,lambda_hat,VT_hat = svd(concatenate((tmp3,tmp4),axis=0))

"""
Obtain best rank-k approximation of [ P  Q1 ]
                                    [ Q2 Q3 ]
"""
tmp5 = concatenate((I_k,zeros((J.shape[0]-I_k.shape[0],k))),axis=0)
new_U = concatenate((tmp5,J),axis=1).dot(U_hat)

#tmp6 = concatenate((V_k,zeros((V_k.shape[0],VT_hat.shape[1]-V_k.shape[1]))),axis=1)
new_rows_cols = VT_hat.shape[1]-V_k.shape[1]
tmp6 = zeros((V_k.shape[0] + new_rows_cols, V_k.shape[1] + new_rows_cols))
tmp6[:V_k.shape[0],:V_k.shape[1]] = V_k
tmp6[V_k.shape[0]:,V_k.shape[1]:] = identity(new_rows_cols)
new_VT = tmp6.dot(VT_hat.transpose()).transpose()
#tmp7 = concatenate((zeros(V_k.shape[]),identity(VT_hat.shape[1]-V_k.shape[1])),axis=1)
step_6 = new_U.dot(diag(lambda_hat)).dot(new_VT)

"""
Obtain best rank-k approximation of [ P  Q1 ]T  [ P  Q1 ]
                                    [ Q2 Q3 ]   [ Q2 Q3 ]
"""

final_sigma = step_6.transpose().dot(step_6)
"""print "U_hat",U_hat,"lambda_hat",lambda_hat,"VT_hat",VT_hat
print "Sigma",Sigma
print "Final Sigma",final_sigma
"""
print norm(Sigma-final_sigma)
