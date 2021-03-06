#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *

#generate data
N = 1000
p = 1
m = 1
X,Y,B_ans = linear_n_Xp_Ym(N,p,m)

#perform the fit
B = linear_regression(X,Y,True)

#see how close we got to the coefficients used for the model
print 'coefficients used to generate test data (answer)'
print B_ans
print 'coefficients from fit'
print B

#Training Error
predicted_Y = np.dot(np.column_stack((X,np.ones(X.shape[0]))),B) #add ones col to X to get intercept contriubtion
mse = mean_squared_error(predicted_Y,Y)
print 'mean squared errors on training data for each element of output vector, Y:'
print mse

#Cross Validation
cvn = 10
print 'performing cross validation CV('+str(cvn)+')'
cvn_result = n_cross_validation(X,Y,linear_regression_train_test,None,cvn)
print 'cross validation errors for each element of output vector, Y:'
print cvn_result

#plot if easy to visualize
if (p == 1) and (m==1):

   fig = plt.figure()
   ax = fig.add_subplot(111) #grab axis object

   ax.scatter(X,Y, s=10, c='b', marker="s", label='Training Data')
   ax.scatter(X,predicted_Y, s=10, c='r', marker="o", label='Predictions')
   plt.legend(loc='upper left');
   plt.show()

