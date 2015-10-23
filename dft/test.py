#! /usr/bin/env python
# coding: utf-8

# In[154]:

import numpy as np
import os
import sys
import math
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()+'/../')
sys.path.append(os.getcwd()+'/../nonlinear_solvers')
sys.path.append(os.getcwd()+'/../stat_tests')
from lasso import lasso



rootdir=os.getcwd()

DatapointEnergies=np.loadtxt(open(rootdir+'/SR_LSDAPW92VV10_w0p3-cx0p15-4258_QZVPPD_250974_Energies.csv'))
DatapointJacobian=np.loadtxt(open(rootdir+'/SR_LSDAPW92VV10_w0p3-cx0p15-4258_QZVPPD_250974_Jacobian.csv'),delimiter=",")
RefValues=np.loadtxt(open(rootdir+'/Reference_New.csv'))
WTrainDiagonal=np.loadtxt(open(rootdir+'/WTrainDiagonal_Attempt26.csv'))
WTotalDiagonal=np.loadtxt(open(rootdir+'/WTotalDiagonal_Attempt26.csv'))


# In[155]:

RefValues.shape


# In[157]:

y = np.array(RefValues - DatapointEnergies)
print y.shape


# In[158]:

W = np.array(WTrainDiagonal)
W_test = np.array(WTotalDiagonal - WTrainDiagonal)


# In[180]:

X = np.array(DatapointJacobian)
#X = np.column_stack((X[:,1:45],X[:,46:90],X[:,91:135] ))
#X = X[:,[1,2,9,10,11,18,19,20,46,47,48,49,50,51,54,55,56,57,58,59,60,63,64,65,66,67,68,69,72,73,74,75,76,77,78,81,82,83,84,85,86,87,91,92,93,94,95,96,99,100,101,102,103,104,105,108,109,110,111,112,113,114,117,118,119,120,121,122,123,126,127,128,129,130,131,132]]
X = X[:,[9,18,27,36,54,63,72,81,99,108,117,126]]
X.shape


# In[181]:

metric = np.dot(np.transpose(X),np.dot(np.diag(W),X))
U,s,V = np.linalg.svd(metric) #NB svals are decreasing
print 'the eigenvalues of the metric'
print s
print(np.finfo(float).eps)
tol = 1.0E-8
n_indep = s.size
for i in s:
    if (i < tol):
        n_indep = n_indep - 1
print 'number of nonzero ',n_indep
if n_indep > 0:
    metric_inv = np.dot(U[:,:n_indep],np.dot(np.diag(s[:n_indep]**-1),np.transpose(U[:,:n_indep])))
else:
    print 'poorly posed problem in linear_regression'
    sys.exit(1)
beta = np.dot(metric_inv,np.dot(np.transpose(X),np.dot(np.diag(W),y)))
#beta = np.dot(np.linalg.inv(metric),np.dot(np.transpose(X),np.dot(np.diag(W),y)))
print beta.shape
print beta


# In[182]:

coef_v_lambda = []
test_v_lambda = []
train_v_lambda = []


# In[189]:

#lambdas = np.linspace(0.000,0.02,211,endpoint=True)
lambdas = np.arange(0.0,0.02,0.0001)
print lambdas


# In[190]:

lasso_obj = lasso(X,y,False,W,True,True)


# In[191]:

lasso_obj.obtain_guess_for_lambda(lambdas[0])


# In[192]:

for l in lambdas:
      converged = lasso_obj.train(l,False)
      if converged:
         y_pred = np.dot(X,lasso_obj.beta)
         err = y-y_pred
         train_rms = math.sqrt(np.dot(err,np.dot(np.diag(W),err))/float(1072))
         test_rms = math.sqrt(np.dot(err,np.dot(np.diag(W_test),err))/float(len(W)-1072))
         coef_v_lambda.append(lasso_obj.beta)
         test_v_lambda.append(test_rms)
         train_v_lambda.append(train_rms)
         print 'lambda = '+str(l)+' train err = '+str(train_rms)+' test err = '+str(test_rms)
         print lasso_obj.beta
      else:
         print 'failed to converge for lambda ='+str(l)
         coef_v_lambda.append(np.zeros(1))
         test_v_lambda.append(-1.0)
         train_v_lambda.append(-1.0)


# In[ ]:



