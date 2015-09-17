#! /usr/bin/env python
import math
import numpy as np

#general notes:
# X has p linear columns (quadratic functions make more)
# Y is n by m
#all X will be relevant for Y
#if you want irrelvant columns in X, add random mats after these calls
#we also pass back B, the answer

#linear_n_Xp_Ym : x1 x2...
def linear_n_Xp_Ym(n,p,m):
#{
   X = np.random.rand(n,p)
   B = (10.0*np.random.rand(p+1,m)) + 1.5 #p+1 to add intercept
   Y = np.dot(np.column_stack((X,np.ones(X.shape[0]))),B) #add a ones column for the intercept
   #add random error epsilon
   Y += np.random.normal(0.0,1.0,(n,m)) # mean 0 var 1 normal draw for epsilon
   
   return X,Y,B
#}


#quadratic_n_Xp_Ym : x1 x2 x1^2 x2^2...
def quadratic_n_Xp_Ym(n,p,m):
#{
   X = np.random.rand(n,p)
   X = np.column_stack((X,np.power(X,2)))
   B = (10.0*np.random.rand(p+p+1,m)) + 1.5 #2p+1 to add intercept
   Y = np.dot(np.column_stack((X,np.ones(X.shape[0]))),B) #add a ones column for the intercept
   #add random error epsilon
   Y += np.random.normal(0.0,1.0,(n,m)) # mean 0 var 1 normal draw for epsilon
   
   return X,Y,B
#}

#quadratic_cross_n_Xp_Ym : x1 x2 x1^2 x1x2 x2^2...
def quadratic_cross_n_Xp_Ym(n,p,m):
#{
   X = np.random.rand(n,p)
   for i in range(p):
      for j in range(i,p):
         X = np.column_stack((X,np.multiply(X[:,i],X[:,j])))
   B = (10.0*np.random.rand(X.shape[1]+1,m)) + 1.5 
   Y = np.dot(np.column_stack((X,np.ones(X.shape[0]))),B) #add a ones column for the intercept
   #add random error epsilon
   Y += np.random.normal(0.0,1.0,(n,m)) # mean 0 var 1 normal draw for epsilon

   return X,Y,B
#}

