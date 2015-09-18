#! /usr/bin/env python
import math
import numpy as np
import scipy.misc

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

def polynomial_n_Xp_Ym(n,p,m,orders,n_relevant_orders,include_cross):
#{
   #orders orders can be any (can skip linear if you want)
   #an intercept is always added
   
   #example
   #  orders = [1,2,3]
   #  n_relevant_orders = [1,1,2]
   #produces data for a model that depends on
   #one first, one second, and two third order terms

   X = np.random.rand(n,p) #we only return the linear predictors
   
   X_relevant = np.zeros((X.shape[0],sum(n_relevant_orders)))
   relevant_pos = 0
   for o in range(len(orders)):
      if (orders[o] <1):
         print 'non-positive order requested. newp'
         sys.exit(1)
      if (n_relevant_orders[o] > 0):
         #we need to pick at least one term for this order
         X_o = poly_order_of_cols(X,orders[o],include_cross)
         columns_chosen = np.random.choice(X_o.shape[1],n_relevant_orders[o],replace=False)
         X_relevant[:,relevant_pos:relevant_pos+n_relevant_orders[o]] = X_o[:,columns_chosen]
         relevant_pos += n_relevant_orders[o]
   B = (10.0*np.random.rand(X_relevant.shape[1]+1,m)) + 1.5 
   Y = np.dot(np.column_stack((X_relevant,np.ones(X_relevant.shape[0]))),B) #add a ones column for the intercept
   #add random error epsilon
   Y += np.random.normal(0.0,1.0,(n,m)) # mean 0 var 1 normal draw for epsilon

   return X,Y
#}

#given columns in X, return a matrix of all columns raised to a set of polynomial orders with or without cross terms
def poly_order_of_cols(X,orders,include_cross):
#{
   if type(orders) is int:
      order_list = [orders]
   else:
      order_list = orders
   
   result = np.ndarray((X.shape[0],0))
   for o in order_list:
      if not type(o) is int:
         print 'Polynomial orders given were not integers'
         sys.exit(1)
      if not include_cross:
         result = np.column_stack((result,np.power(X,o)))
      else:
         #we need to generate cross terms as well
         p = X.shape[1]
         #there are (o + p -1) choose (o) such terms
         #put o indistinguishable objects in p distinguishable boxes
         power_row_vectors = o_obj_b_boxes(o,p)
         n_combinations = scipy.misc.comb(o+p-1,o,exact=True)
         if not (n_combinations == power_row_vectors.shape[0]):
            print 'error occurred in generating combinations of powers'
            print 'combinations produced by o_obj_b_boxes: ',power_row_vectors.shape[0]
            print 'based on binomial coef formula it should be: ',n_combinations
            sys.exit(1)
         to_add = np.zeros((X.shape[0],n_combinations))
         for i in range(n_combiantions):
            temp = np.ones(X.shape[0])
            for c in range(p):
               if (power_row_vector[i,c] > 0): 
                  if (power_row_vector[i,c] == 1):
                     #no need to take the column to a power
                     temp = np.multiply(temp,X[:,c])
                  else:
                     temp = np.multiply(temp,np.power(X[:,c],power_row_vector[i,c]))
            to_add[:,i] = temp
         result = np.column_stack((result,to_add))
   return result
#}

def o_obj_b_boxes(o,b):
#{
   #putting o indistinguishable objects in b distinguishable boxes
   #returns a matrix with rows corresponding to the different ways
   #and columns 
   if (o<0) or (b<1):
      print 'something went horribly wrong in o_obj_b_boxes' 
      if (o<0):
         print 'negative number of objects passed!'
      if (b<1):
         print 'non-positive number of boxes passed!'
      sys.exit(1)

   if (b == 1):
      #there is only one box. Put everything in that box!
      ans = np.ndarray((1,b),dtype=np.int)
      ans.fill(o)
      return ans
   elif (o == 0):
      #there are no objects. Return empty boxes!
      ans = np.ndarray((1,b),dtype=np.int)
      ans.fill(0)
      return ans
   else:
      #the answer isn't trivial yet...
      #decrease the number of boxes by fixing the population 
      #of the leftmost box and then call again on the smaller problem
      ans = np.ndarray((0,b),dtype=np.int) 
      for f in reversed(range(o+1)):  #o o-1 o-2 ... 0  --- f for fixed
         sub_ans_f = o_obj_b_boxes(o-f,b-1)
         #prepend our fixed value
         fixed_add =  np.ndarray((sub_ans_f.shape[0],1),dtype=np.int) 
         fixed_add.fill(f)
         sub_ans_f = np.column_stack((fixed_add,sub_ans_f))
         ans =  np.vstack((ans,sub_ans_f))
      return ans
#}



