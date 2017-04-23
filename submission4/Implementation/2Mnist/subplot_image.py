from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt

W_ppca = load('ppca_model_W.npy')
Xn_M_ppca = load('ppca_model_Xn_M.npy')
W_ppca = W_ppca[0,:,:]
Xn_M_ppca = Xn_M_ppca[0,:,:]
ppca = W_ppca.dot(inv(W_ppca.T.dot(W_ppca))).dot(Xn_M_ppca)

j = [100,200,300,400]
k = [221,222,223,224]
fig = plt.figure(1)
plt.suptitle('Generated Mnist Image of \'1\' at nth iteration')
for i,h in zip(j,k):

    W = load('generator_mnist_'+str(i)+'_W.npy')
    Xn_M = load('generator_mnist_'+str(i)+'_Xn_M.npy')
    W = W[0,:,:,0]
    Xn_M = Xn_M[0,:,:,0]
    rd = W.dot(inv(W.T.dot(W))).dot(Xn_M)
    rd = W.dot(inv(W.T.dot(W))).dot(Xn_M)

    plt.subplot(h)
    plt.imshow(rd, cmap=plt.cm.Greys_r)
    plt.title(str(i)+'th iteration')
    print i,": ",norm(ppca-rd)
plt.show()
fig.savefig('Subplot.png', dpi=fig.dpi,bbox_inches='tight')
