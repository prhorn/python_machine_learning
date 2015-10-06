#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *

#generate data
n=100
p=3
m=1
max_order = 2
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

#Now lets try a regression tree
print 'computing training error for Unpruned Decision Tree'
train_udt = decision_tree_train_test(X,Y,X,Y,(False,10))
print 'unpruned decision tree training mse is ',train_udt

print 'performing cross validation CV('+str(cvn)+') for Unpruned Decision Tree'
cvn_udt = n_cross_validation(X,Y,decision_tree_train_test,(False,10),cvn)
print 'unpruned decision tree cvn mse is ',cvn_udt

print 'performing cross validation CV('+str(cvn)+') for Pruned Decision Tree'
alpha_values = np.linspace(1.0,20.0,num=lambdas.size,endpoint=True) #so that we can put it on the same plot easily
errors_for_alpha = []
for a in alpha_values:
   cvn_pdt = n_cross_validation(X,Y,pruned_decision_tree_train_test,(False,10,a),cvn)
   print 'pruned decision tree (alpha='+str(a)+') cvn mse is '+str(cvn_pdt[0])
   errors_for_alpha.append(cvn_pdt[0])


if (m==1):
   fig = plt.figure()
   ax = fig.add_subplot(111)
   lr_vs_lambda = np.zeros(lambdas.size)
   lr_vs_lambda.fill(cvn_lr[0])
   ax.scatter(lambdas,lr_vs_lambda,s=10,c='b',marker="s",label="LR CVn")
   ax.scatter(lambdas,rr_vs_lambda,s=10,c='r',marker="s",label="RR CVn vs lambda")
   udt_train_vs_lambda = np.zeros(lambdas.size)
   udt_train_vs_lambda.fill(train_udt[0])
   ax.scatter(lambdas,udt_train_vs_lambda,s=10,c='g',marker="s",label="Decision Tree Train")
   udt_cvn_vs_lambda = np.zeros(lambdas.size)
   udt_cvn_vs_lambda.fill(cvn_udt[0])
   ax.scatter(lambdas,udt_cvn_vs_lambda,s=10,c='m',marker="s",label="Decision Tree CVn")
   ax.scatter(lambdas,errors_for_alpha,s=10,c='y',marker="s",label="Pruned Decision Tree CVn vs alpha")
   plt.legend(loc='best');
   plt.show()
