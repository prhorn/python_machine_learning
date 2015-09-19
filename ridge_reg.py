#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *

#generate data
n=1000
p=5
m=1
max_order = 3
orders = range(1,max_order+1)
n_relevant_orders = np.ndarray(len(orders),dtype=np.int)
n_relevant_orders.fill(p)
include_cross_terms = True
X,Y = polynomial_n_Xp_Ym(n,p,m,orders,n_relevant_orders,include_cross_terms)

#we are going to try to find the sum(n_releant_orders) 
#important predictors. first compute all X possible products
X_aug = poly_order_of_cols(X,orders,include_cross_terms)

#Cross Validation
cvn = 10
print 'performing cross validation CV('+str(cvn)+') for linear regression'
cvn_lr = n_cross_validation(X_aug,Y,linear_regression_train_test,None,cvn)
print 'linear regression cvn mse is ',cvn_lr

#Now we will compute cvn for ridge regression with a number of lambdas
rr_vs_lambda = []
lambdas = np.linspace(0.0,2.0,20,endpoint=False)
for l in lambdas:
   print 'performing cross validation CV('+str(cvn)+') for ridge regression('+str(l)+')'
   cvn_rr = n_cross_validation(X_aug,Y,ridge_regression_train_test,l,cvn)
   print 'ridge regression lambda is '+str(l)+' and cvn mse is ',cvn_rr
   rr_vs_lambda.append(cvn_rr[0])

if (m==1):
   fig = plt.figure()
   ax = fig.add_subplot(111)
   lr_vs_lambda = np.zeros(lambdas.size)
   lr_vs_lambda.fill(cvn_lr[0])
   ax.scatter(lambdas,lr_vs_lambda,s=10,c='b',marker="s",label="LR")
   ax.scatter(lambdas,rr_vs_lambda,s=10,c='r',marker="o",label="RR")
   plt.legend(loc='upper right');
   plt.show()

