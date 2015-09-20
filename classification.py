#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *
from sklearn.lda import LDA

n=100
p=2
m=5
X,Y = classification_gaussians(n,p,m,False,True,True)
#mean = np.array([40,60])
#cov = np.array([100,30,30,140]).reshape(2,2)
#x,y = np.random.multivariate_normal(mean,cov,5000).T
#plt.plot(x,y,'x'); plt.axis('equal'); plt.show()

print 'computing error on training data for Linear Discriminant Analaysis'
train_err_LDA = lin_quad_discriminant_analysis_train_test(X,Y,X,Y,(True,False,0.5,0.5))
print 'training data  error rate for LDA is  '+str(train_err_LDA[0])

print 'computing error on training data using scikitlearn LDA'
clf = LDA()
clf.fit(X,Y.reshape((Y.size)))
Y_sci_LDA_predict = clf.predict(X)
err_sci_lda = classification_error_rate(Y_sci_LDA_predict.reshape((Y_sci_LDA_predict.size,1)),Y) 
print 'training data  error rate for LDA is  '+str(err_sci_lda[0])

print 'computing cross validation error for Linear Discriminant Analaysis'
cv_LDA = n_cross_validation(X,Y,lin_quad_discriminant_analysis_train_test,(True,False,0.5,0.5),10)
print 'cross validation classification error rate for LDA is  '+str(cv_LDA[0])

k_max = 20
k_values = range(1,k_max+1)
errors_for_k = np.zeros(k_max)
#use cross validation to choose k
for k in k_values:
   cv = n_cross_validation(X,Y,k_nearest_neighbors_classification_train_test,k,10)
   print 'cross validation classification error rate for kNN k='+str(k)+' is  '+str(cv[0])
   errors_for_k[k-1] = cv[0]

fig = plt.figure()
ax = fig.add_subplot(111) #grab axis object
ax.scatter(k_values,errors_for_k, s=10, c='r', marker="s", label='kNN Classification Error Rate vs k')
LDA_const = np.zeros(len(k_values))
LDA_const.fill(cv_LDA[0])
ax.scatter(k_values,LDA_const, s=10, c='b', marker="s", label='LDA Error Rate')
plt.legend(loc='best')
plt.show()
