import numpy as np

import numpy.linalg as LA

import scipy.linalg as sciLA

import numpy.matlib as npML

def ppca_mv(Ye, d, dia):
    #
    # implements probabilistic PCA for data with missing values,
    # using a factorizing distribution over hidden states and hidden observations.
    #
    #  - The entries in Ye that equal NaN are assumed to be missing. -
    #
    # [C, ss, M, X, Ye ] = ppca_mv(Y,d,dia)
    #
    # Y   (N by D)  N data vectors
    # d   (scalar)  dimension of latent space
    # dia (binary)  if 1: printf objective each step
    #
    # ss  (scalar)  isotropic variance outside subspace
    # C   (D by d)  C*C' +I*ss is covariance model, C has scaled principal directions as cols.
    # M   (D by 1)  data mean
    # X   (N by d)  expected states
    # Ye  (N by D)  expected complete observations (interesting if some data is missing)
    #
    # J.J. Verbeek, 2006. http://lear.inrialpes.fr/~verbeek
    #

    N, D = Ye.shape # N observations in D dimensions
    threshold   = 10**(-4)     # minimal relative change in objective funciton to continue
    hidden = np.isnan(Ye)
    missing = np.count_nonzero(hidden)

    M = np.zeros((1,D),dtype=np.float64)  # compute data mean and center data
    if missing:
        for i in range(0,D):
            M[0,i] = np.mean(Ye[~hidden[:,i],i])
    else:
        M = np.mean(Ye)
    print Ye.shape
    print npML.repmat(M, N, 1).shape
    print N
    Ye = Ye - npML.repmat(M, N, 1)

    if missing:
        Ye[hidden] = 0

    # =======     Initialization    ======
    C     = np.random.randn(D,d)
    CtC   = C.transpose().dot(C)
    X     = Ye.dot(C).dot(LA.inv(CtC))
    recon = X.dot(C.transpose())
    recon[hidden] = 0
    ss    = np.sum(np.sum(np.square(recon-Ye))) / (N*D-missing)

    count = 1
    old   = np.inf
    while count:          #  ============ EM iterations  ==========

        Sx = LA.inv( np.eye(d) + CtC/ss)    # ====== E-step, (co)variances   =====
        ss_old = ss
        if missing:
            proj = X.dot(C.transpose())
            Ye[hidden] = proj[hidden]
        X = Ye.dot(C.dot(Sx/ss))          # ==== E step: expected values  ====

        SumXtX = X.transpose().dot(X)                  # ======= M-step =====
        C      = Ye.transpose().dot(X).dot(LA.inv(SumXtX + N*Sx ))
        CtC    = C.transpose().dot(C)
        ss     = ( np.sum(np.sum( np.square(X.dot(C.transpose())-Ye) )) + N*np.sum(np.sum(np.multiply(CtC,Sx))) + missing*ss_old ) /(N*D)

        objective = N*D + N*(D*np.log(ss) + np.trace(Sx)- np.log(LA.det(Sx)) ) + np.trace(SumXtX) - missing*np.log(ss_old)

        rel_ch    = np.absolute( 1 - objective / old )
        old       = objective

        count = count + 1
        if rel_ch < threshold and count > 5:
            count = 0
        if dia:
            print('Objective:  %.2f    relative change: %.5f \n')  % (objective, rel_ch)

                      #  ============ EM iterations  ==========


    C = sciLA.orth(C)
    vals,vecs = LA.eig(np.cov(Ye.dot(C).transpose()))

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:,order]

    C = C.dot(vecs)
    X = Ye.dot(C)

    # add data mean to expected complete data
    Ye = Ye + npML.repmat(M, N, 1)
    return C, ss, M, X,Ye
