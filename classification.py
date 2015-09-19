#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *

n=50
p=2
m=5
X,Y = classification_gaussians(n,p,m,False,True)
#mean = np.array([40,60])
#cov = np.array([100,30,30,140]).reshape(2,2)
#x,y = np.random.multivariate_normal(mean,cov,5000).T
#plt.plot(x,y,'x'); plt.axis('equal'); plt.show()

k_max = 50
k_values = range(1,k_max+1)
errors_for_k = np.zeros(k_max)
#use cross validation to choose k
for k in k_values:
   cv = n_cross_validation(X,Y,k_nearest_neighbors_classification_train_test,k,10)
   print 'cross validation classification error rate for kNN k='+str(k)+' is  '+str(cv[0])
   errors_for_k[k-1] = cv[0]

fig = plt.figure()
ax = fig.add_subplot(111) #grab axis object
ax.scatter(k_values,errors_for_k, s=10, c='b', marker="s", label='kNN Classification Error Rate vs k')
plt.legend(loc='best')
plt.show()

