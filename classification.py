#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *

n=50
p=2
m=5
X,Y = classification_gaussians(n,p,m,False,False,True)

print 'computing error on training data for Linear Discriminant Analaysis'
train_err_LDA = lin_quad_discriminant_analysis_train_test(X,Y,X,Y,(True,False,0.5))
print 'training data  error rate for LDA is  '+str(train_err_LDA[0])

print 'computing cross validation error for Linear Discriminant Analaysis'
cv_LDA = n_cross_validation(X,Y,lin_quad_discriminant_analysis_train_test,(True,False,0.5),10)
print 'cross validation classification error rate for LDA is  '+str(cv_LDA[0])

print 'computing error on training data for Quadratic Discriminant Analaysis'
train_err_QDA = lin_quad_discriminant_analysis_train_test(X,Y,X,Y,(False,False,0.5))
print 'training data  error rate for QDA is  '+str(train_err_QDA[0])

print 'computing cross validation error for Quadratic Discriminant Analaysis'
cv_QDA = n_cross_validation(X,Y,lin_quad_discriminant_analysis_train_test,(False,False,0.5),10)
print 'cross validation classification error rate for QDA is  '+str(cv_QDA[0])

print 'computing cross validation error for Gaussian Naive Bayes'
cv_gnb = n_cross_validation(X,Y,gaussian_naive_bayes_train_test,(False),10)
print 'cross validation classification error rate for Gaussian Naive Bayes is  '+str(cv_gnb[0])

print 'computing training error for Unpruned Decision Tree'
train_err_udt = decision_tree_train_test(X,Y,X,Y,(True,3))
print 'training data  error rate for Unpruned Decision Tree is  '+str(train_err_udt[0])

print 'computing cross validation error for Unpruned Decision Tree'
cv_udt = n_cross_validation(X,Y,decision_tree_train_test,(True,3),10)
print 'cross validation classification error rate for Unpruned Decision Tree is  '+str(cv_udt[0])

k_max = 20
k_values = range(1,k_max+1)
errors_for_k = np.zeros(k_max)
#use cross validation to choose k
print 'computing cross validation error for kNN'
for k in k_values:
   cv = n_cross_validation(X,Y,k_nearest_neighbors_classification_train_test,k,10)
   print 'cross validation classification error rate for kNN k='+str(k)+' is  '+str(cv[0])
   errors_for_k[k-1] = cv[0]

print 'computing training error for Regularized Discriminant Analaysis alpha=0.5'
train_err_RDA = lin_quad_discriminant_analysis_train_test(X,Y,X,Y,(False,True,0.5))
print 'training data  error rate for RDA alpha=0.5 is  '+str(train_err_RDA[0])

alpha_values = np.linspace(0.05,0.95,num=k_max,endpoint=True) #so that we can put it on the same plot easily
errors_for_alpha = []
#use cross validation to choose alpha
print 'computing cross validation error for RDA'
for a in alpha_values:
   cva = n_cross_validation(X,Y,lin_quad_discriminant_analysis_train_test,(False,True,a),10)
   print 'cross validation classification error rate for RDA alpha='+str(a)+' is  '+str(cva[0])
   errors_for_alpha.append(cva[0])

#print 'computing cross validation error for Pruned Decision Tree'
#alpha_values_tree = np.linspace(0.5,10.0,num=k_max,endpoint=True) #so that we can put it on the same plot easily
#errors_for_alpha_tree = []
#for a in alpha_values_tree:
#   cv_pdt = n_cross_validation(X,Y,pruned_decision_tree_train_test,(True,5,a),10)
#   print 'cross validation classification error rate for Pruned Decision Tree (alpha='+str(a)+') is  '+str(cv_pdt[0])
#   errors_for_alpha_tree.append(cv_pdt[0])

fig = plt.figure()
ax = fig.add_subplot(111) #grab axis object
ax.scatter(k_values,errors_for_k, s=10, c='r', marker="s", label='kNN Classification Error Rate vs k')
LDA_const = np.zeros(len(k_values))
LDA_const.fill(cv_LDA[0])
ax.scatter(k_values,LDA_const, s=10, c='b', marker="s", label='LDA Error Rate')
QDA_const = np.zeros(len(k_values))
QDA_const.fill(cv_QDA[0])
ax.scatter(k_values,QDA_const, s=10, c='g', marker="s", label='QDA Error Rate')
ax.scatter(k_values,errors_for_alpha, s=10, c='m', marker="s", label='RDA Classification Error Rate vs even alpha(0-1)')
gnb_const = np.zeros(len(k_values))
gnb_const.fill(cv_gnb[0])
ax.scatter(k_values,gnb_const, s=10, c='y', marker="s", label='Gaussian Naive Bayes Error Rate')
plt.legend(loc='best')
plt.show()
