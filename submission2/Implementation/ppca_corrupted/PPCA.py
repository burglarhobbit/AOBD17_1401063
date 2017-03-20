# Principal Component Analysis based on Expectation Maximization Algorithm.

# Syntax for defining a function.
import numpy as np
from numpy.linalg import inv

def em_PPCA(t, k):
    if k < 1 | k > t.shape[1]:
        print('Number of Principle Components must be integer, >0, <dim')

    # Number of iteration.
    ite = 20

    # Finding the height and the width of the data matrix.
    [height, width] = t.shape
    t_mean = np.zeros((len(t),1))
    # Finding the mean value of the observed data vectors.
    # t_mean = (sum(t(:,1:width)')')/width;
    # print(t[:, 0:width].shape)
    for i in range(width):
        t_mean[i] = (np.sum(t[i,:].T).T) / width

    # t_mean = 0
    # for i in range(width)
    #     t_mean[i] = (np.sum(t[:, i].T).T) / width

    # Normalize the data matrix.

    t = t - np.dot((t_mean) , np.ones((1, width)))
    # print(t)
    # Initially w and sigma square will be randomly selected.

    W = np.random.standard_normal((height, k))
    #print(W.shape)
    sigma_square = np.random.standard_normal((1, 1))
    #print(sigma_square)

    print('EM algorithm is running....Please Wait.......')

    for i in range(ite):
        #print(W)
        #print(sigma_square)
        # According to the equation: M = W'W + Sigma^2*I
        M = np.dot(W.T,W) + (sigma_square * np.eye(k, k))

        # Find inverse of M
        inv_M = inv(M)

        # Expected Xn
        Xn = np.zeros((k, width))
        Xn_Xn_T = np.zeros((k, k))

        for i in range(width):
            Xn[:, i] = np.dot(np.dot((inv_M), W.T) ,t[:,i])
            # Find Expected of XnXn'
            Xn_Xn_T = Xn_Xn_T + (sigma_square * (inv_M) + np.dot( (Xn[:, i].reshape(len(Xn),1)) , (Xn[:, i].reshape(len(Xn),1).T)))

        #print(Xn)
        # Taking the old value of W
        old_W = W

        temp1 = np.zeros((height, k))

        # print(t.shape)

        for i in range(width):

            temp1 = temp1 + np.dot(t[:,i].reshape(height,1) , ((Xn[:,i]).reshape(len(Xn),1).T))

            # Taking the new value of W
        W = np.dot(temp1 , inv(Xn_Xn_T))

        #print(W)
        sum11 = 0

        for i in range(width):

            temp2 = sigma_square * inv_M + np.dot( (Xn[:,i].reshape(len(Xn),1)) , (Xn[:,i].reshape(len(Xn),1).T) )
            sum11 = sum11 + ((np.linalg.norm(t[:,1]) ** 2) - np.dot(np.dot((2 * (Xn[:,i].reshape(len(Xn),1).T)) ,(W.T)) , (t[:,i].reshape(height,1)) ) + np.trace((np.dot(np.dot(temp2 , (W.T)) , W))))

        #print(temp2)
        sigma_square = sum11 / (width * height)

    print('EM Algorithm Completed. W is created. Press enter to continue')
    print(sigma_square)
    M = np.dot((W.T) , W) + sigma_square * np.eye(k,k)
    In_M = inv(M)

    Xn = np.dot(np.dot((In_M),(W.T)) , t)

    print('Principal Component are ready.. press enter to continue')
    # print(sigma_square)
    # print(W)
    return W, sigma_square, Xn, t_mean, M
