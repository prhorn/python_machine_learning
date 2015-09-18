#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *


#generate data
n=1000
p=1
m=1
X,Y,B_ans = linear_n_Xp_Ym(n,p,m)

k_max = 100
k_values = range(1,k_max+1)
errors_for_k = np.zeros((k_max,m))
#use cross validation to choose k
for k in k_values:
   cv = n_cross_validation(X,Y,k_nearest_neighbors_train_test,k,10)
   print 'cross validation errors for kNN k ='+str(k)
   print cv
   errors_for_k[k-1,:] = cv

#compare to linear regression
cv_lr = n_cross_validation(X,Y,linear_regression_train_test,None,10)
print 'cross validation errors for linear regression:'
print cv_lr
errors_for_lr = np.zeros((k_max,m))
for i in range(m):
   errors_for_lr[:,i].fill(cv_lr[i])

fig = plt.figure()
ax = fig.add_subplot(111) #grab axis object
for i in range(m):
   ax.scatter(k_values,errors_for_k[:,i], s=10, c='b', marker="s", label='KNN out '+str(i))
   ax.scatter(k_values,errors_for_lr[:,i], s=10, c='r', marker="o", label='LR out '+str(i))
plt.legend(loc='upper right');
plt.show()
