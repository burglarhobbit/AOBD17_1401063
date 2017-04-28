import numpy as np
from sklearn.linear_model import SGDRegressor as SGD
from numpy import *
import profile_read_json as prj

np.random.seed(0)

user_job, user_skill, skillset_numeric, skillsetjob_numeric = prj.get_model()

jobs = user_job.shape[1]
features = user_skill.shape[1]
parameters = zeros((features,jobs))

for i in range(user_job.shape[1]):
    #print i
    index = user_job[:,i]==1
    #print index
    y = user_job[:,i][index]
    X = user_skill[index]
    if True in index:
        #print y.shape
        #print X.shape
        clf = SGD().fit(X,y)
        coefs = clf.coef_
        #print coefs.shape
        parameters[:,i] = coefs
    else:
        pass
#print parameters

threshold = 0.2
userid = 100
# nans for user 1

user_nan = isnan(user_job[userid])
for j,i in enumerate(user_nan):
    #print i
    # j is job id
    if i:
        theta_j = parameters[:,j]
        feature_i = user_skill[userid]
        y_pred = theta_j.T.dot(feature_i.T)
        #print y_pred
        if y_pred<threshold:
            for k in skillsetjob_numeric.keys():
                if skillsetjob_numeric[k] == j:
                    print "\nPreferred Job:",k
            # predict skills
            non_zero_theta_j = theta_j>0
            zero_feature_i = feature_i==0
            for q,l,m in zip(range(len(non_zero_theta_j)),non_zero_theta_j,zero_feature_i):
                #print q,l,m
                if l == True and m == True:
                    for k in skillset_numeric.keys():
                        if skillset_numeric[k] == q:
                            print "Prescribed Skills: ",k
