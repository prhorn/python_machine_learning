#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *
from lasso import lasso

#generate data
n=4000
p=7
m=1
max_order = 2
np.random.seed(11)
orders = range(1,max_order+1)
n_relevant_orders = np.ndarray(len(orders),dtype=np.int)
n_relevant_orders.fill(p)
#n_relevant_orders.fill(0) #mean model
include_cross_terms = True
X,Y = polynomial_n_Xp_Ym(n,p,m,orders,n_relevant_orders,include_cross_terms)

#we are going to try to find the sum(n_releant_orders) 
#important predictors. first compute all X possible products
X_aug = poly_order_of_cols(X,orders,include_cross_terms)

print 'cols of X_aug ',X_aug.shape[1]

##stats
Y_fix = Y - np.sum(Y)/float(Y.size)
B = linear_regression(X_aug,Y_fix[:,0],True)
r_sq = lin_reg_coefficient_of_determination(X_aug,Y_fix[:,0],B)
print 'r_squared for brute force fit ',r_sq
lin_reg_statistics(X_aug,Y_fix[:,0],B)
print 'full linreg coefs'
print B

#Cross Validation
#cvn = 10
#print 'performing cross validation CV('+str(cvn)+') for linear regression'
#cvn_lr = n_cross_validation(X_aug,Y,linear_regression_train_test,None,cvn)
#print 'linear regression cvn mse is ',cvn_lr

#Now we will compute cvn for LASSO  with a number of lambdas
if m==1:
   lasso_train_vs_lambda = []
   lasso_test_vs_lambda = []
   lambdas = np.linspace(0.0000,1000.0,11,endpoint=True)
   n_test = int(X_aug.shape[0]/5)
   Y_shift = Y - np.sum(Y)/float(Y.size)
   X_aug_test = X_aug[:n_test,:]
   Y_shift_test = Y_shift[:n_test,:]
   X_aug_train = X_aug[n_test:,:]
   Y_shift_train = Y_shift[n_test:,:]
   lasso_obj = lasso(X_aug_train,Y_shift_train,False,np.ones(len(Y_shift_train)),True,True)
   lasso_obj.obtain_guess_for_lambda(lambdas[0])
   for l in lambdas:
      converged = lasso_obj.train(l,False)
      if converged:
         predicted_Y_shift_train = np.dot(X_aug_train,lasso_obj.beta) 
         predicted_Y_shift_test = np.dot(X_aug_test,lasso_obj.beta) 
         mse_train = mean_squared_error(predicted_Y_shift_train.reshape((len(predicted_Y_shift_train),1)),Y_shift_train) 
         mse_test = mean_squared_error(predicted_Y_shift_test.reshape((len(predicted_Y_shift_test),1)),Y_shift_test) 
         print 'lambda = '+str(l)+' train mse = '+str(mse_train)+' test mse = '+str(mse_test)
         lasso_train_vs_lambda.append(mse_train)
         lasso_test_vs_lambda.append(mse_test)
         print 'converged coefs '
         print lasso_obj.beta
      else:
         print 'failed to converge for lambda ='+str(l)
         lasso_train_vs_lambda.append(-1.0)
         lasso_test_vs_lambda.append(-1.0)
   print 'SUMMARY:'
   print 'lambda        train          test'
   for l in range(len(lambdas)):
      print str(lambdas[l])+'    '+str(lasso_train_vs_lambda[l])+'      '+str(lasso_test_vs_lambda[l])
   
   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.scatter(lambdas,lasso_train_vs_lambda,s=10,c='b',marker="s",label="LASSO train")
   ax.scatter(lambdas,lasso_test_vs_lambda,s=10,c='r',marker="s",label="LASSO test")
   plt.legend(loc='best');
   plt.show()



#Now we will compute cvn for ridge regression with a number of lambdas
#rr_vs_lambda = []
#lambdas = np.linspace(0.0,2.0,20,endpoint=False)
#for l in lambdas:
#   print 'performing cross validation CV('+str(cvn)+') for ridge regression('+str(l)+')'
#   cvn_rr = n_cross_validation(X_aug,Y,ridge_regression_train_test,l,cvn)
#   print 'ridge regression lambda is '+str(l)+' and cvn mse is ',cvn_rr
#   rr_vs_lambda.append(cvn_rr[0])

#Now lets try a regression tree
#print 'computing training error for Unpruned Decision Tree'
#train_udt = decision_tree_train_test(X,Y,X,Y,(False,10))
#print 'unpruned decision tree training mse is ',train_udt

#print 'performing cross validation CV('+str(cvn)+') for Unpruned Decision Tree'
#cvn_udt = n_cross_validation(X,Y,decision_tree_train_test,(False,10),cvn)
#print 'unpruned decision tree cvn mse is ',cvn_udt

#print 'performing cross validation CV('+str(cvn)+') for Pruned Decision Tree'
#alpha_values = np.linspace(1.0,20.0,num=lambdas.size,endpoint=True) #so that we can put it on the same plot easily
#errors_for_alpha = []
#for a in alpha_values:
#   cvn_pdt = n_cross_validation(X,Y,pruned_decision_tree_train_test,(False,10,a),cvn)
#   print 'pruned decision tree (alpha='+str(a)+') cvn mse is '+str(cvn_pdt[0])
#   errors_for_alpha.append(cvn_pdt[0])


#if (m==1):
#   fig = plt.figure()
#   ax = fig.add_subplot(111)
#   lr_vs_lambda = np.zeros(lambdas.size)
#   lr_vs_lambda.fill(cvn_lr[0])
#   ax.scatter(lambdas,lr_vs_lambda,s=10,c='b',marker="s",label="LR CVn")
#   ax.scatter(lambdas,rr_vs_lambda,s=10,c='r',marker="s",label="RR CVn vs lambda")
#   udt_train_vs_lambda = np.zeros(lambdas.size)
#   udt_train_vs_lambda.fill(train_udt[0])
#   ax.scatter(lambdas,udt_train_vs_lambda,s=10,c='g',marker="s",label="Decision Tree Train")
#   udt_cvn_vs_lambda = np.zeros(lambdas.size)
#   udt_cvn_vs_lambda.fill(cvn_udt[0])
#   ax.scatter(lambdas,udt_cvn_vs_lambda,s=10,c='m',marker="s",label="Decision Tree CVn")
#   #ax.scatter(lambdas,errors_for_alpha,s=10,c='y',marker="s",label="Pruned Decision Tree CVn vs alpha")
#   plt.legend(loc='best');
#   plt.show()

