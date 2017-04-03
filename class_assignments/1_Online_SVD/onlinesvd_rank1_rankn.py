import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

row = 15
col = 10
for i in range(10,20):
    c = 4
    col = i
    r = int(np.sqrt(min(row,col)))
    #print "r:",r
    A = np.arange(1,row*col+1,1)
    A = A.reshape(row,col)

    U, s, VT = LA.svd(A, full_matrices=True)

    S = np.zeros((r,r), dtype=np.float64)
    S[:r,:r] = np.diag(s[:r])
    U = U[:,:r]
    VT = VT[:r,:]

    a = np.random.randint(1,15,size=(row,c))
    b = np.random.randint(1,15,size=(col,c))

    # QR
    I_p = np.identity(row)
    U_UT = U.dot(U.transpose())

    I_q = np.identity(col)
    V_VT = VT.transpose().dot(VT)

    P,Ra = LA.qr(np.subtract(I_p,U_UT).dot(a))
    Q,Rb = LA.qr(np.subtract(I_q,V_VT).dot(b))

    k_matrix2_00 = np.dot(U.transpose(),a)
    k_matrix2_01 = np.concatenate((k_matrix2_00,Ra),axis=0)
    k_matrix2_10 = np.dot(VT,b)
    k_matrix2_11 = np.concatenate((k_matrix2_10,Rb),axis=0)

    k_matrix2 = np.dot(k_matrix2_01,k_matrix2_11.transpose())
    k_matrix1 = np.zeros(k_matrix2.shape,dtype=np.float64)
    k_matrix1[:r,:r] = S

    K = np.add(k_matrix1,k_matrix2)

    U_1, s_1, VT_1 = LA.svd(K, full_matrices=True)
    s_1 = np.diag(s_1)
    new_U = np.concatenate((U,P),axis=1).dot(U_1)
    #print new_U.shape
    new_VT = np.concatenate((VT.transpose(),Q),axis=1).dot(VT_1.transpose()).transpose()

    new_A = np.add(A,a.dot(b.transpose()))
    com_A = new_U.dot(s_1).dot(new_VT)
    Error_n = np.subtract(new_A,com_A)
    errn = LA.norm(Error_n)
    print "RANK-%d:" %(col)
    print "ERROR:",errn

# RANK-1 MODIFICATIONS
U, s, VT = LA.svd(A, full_matrices=True)
S = np.zeros((r,r), dtype=np.float64)
S[:r,:r] = np.diag(s[:r])
U = U[:,:r]
VT = VT[:r,:]

c = 1
rank = 1
a = np.random.randint(1,15,size=(row,c))
b = np.zeros((col,c),dtype=np.float64)
#print b[-rank:,:]
b[-rank:,:] = np.random.randint(1,15,size=(rank,c))
#print b.shape
#print a.dot(b.transpose())
m = U.transpose().dot(a)
p = np.subtract(a,U.dot(m))
Ra = np.zeros((1,1),dtype=np.float64)
Ra[0,0] = LA.norm(p) # highest eigenvalue
P = p/Ra

n = VT.dot(b)
q = np.subtract(b,VT.transpose().dot(n))
Rb = np.zeros((1,1),dtype=np.float64)
Rb[0,0] = LA.norm(q) # highest eigenvalue
Q = q/Rb
#print m.shape,Ra.shape
k_matrix2_0 = np.concatenate((m,Ra),axis=0)
k_matrix2_1 = np.concatenate((n,Rb),axis=0).transpose()

k_matrix2 = np.dot(k_matrix2_0,k_matrix2_1)
k_matrix1 = np.zeros(k_matrix2.shape,dtype=np.float64)
k_matrix1[:r,:r] = S

K = np.add(k_matrix1,k_matrix2)

U_1, s_1, VT_1 = LA.svd(K, full_matrices=True)
s_1 = np.diag(s_1)
new_U = np.concatenate((U,P),axis=1).dot(U_1)
#print new_U.shape
new_VT = np.concatenate((VT.transpose(),Q),axis=1).dot(VT_1.transpose()).transpose()

new_A = np.add(A,a.dot(b.transpose()))
com_A = new_U.dot(s_1).dot(new_VT)
Error_1 = np.subtract(new_A,com_A)
err1 = LA.norm(Error_1)
print "\nRANK-1",err1

plt.plot([1, col],[err1, errn])
plt.show()
