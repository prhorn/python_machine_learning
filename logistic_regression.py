import numpy as np
import math
import sys
import os
sys.path.append(os.getcwd()+'/nonlinear_solvers/')
from l_bfgs import l_bfgs

#Logistic regression trained based on
#multinomial maximum likelihood

class logistic_regression:
#{
   #X          --    our training predictors with an additional column of
   #                 of 1's added to take care of intercepts.
   #                 sorted so that obs with y=0 first, y=1 second...
   #
   #N          --    number of observations, rows of X and len(y)
   #
   #m          --    cols of X, number of predictors plus 1 for intercept
   #
   #y          --    our training classifications of each of the data points
   #                 coding is assumed 0,1,2,3,K-1
   #
   #K          --    number of categories for classification
   #
   #obs_cat    --    number of observations in each category
   #                 once rows of X are sorted, this enables us to easily grab
   #                 X values for all observations in a certain category
   #
   #B          --    the m by K-1 matrix of coefficients (plus intercepts, last row)
   #B_origin   --
   #

   def __init__(self,X_,y_,K_):
   #{
      self.X = np.column_stack((np.array(X_),np.ones(X_.shape[0])))
      self.y = np.reshape(y_,len(y_))

      self.K = K_
      self.N = self.X.shape[0] #rows of X, number of observations
      self.m = self.X.shape[1] #cols of X, number of covariates
      if not (self.y.size == self.N):
         print "number of observations (rows) in X and y did not match in logistic regression constructor"
         sys.exit(1)
      
      #now lets work on sorting our input data by category 
      sort_indices = np.argsort(self.y)
      self.y = self.y[sort_indices]
      self.X = self.X[sort_indices,:]
      self.obs_cat = np.bincount(self.y)
      if not (self.obs_cat.size == self.K):
         print 'not all K categories had observations. not dealing with this case ATM'
         sys.exit(1)

      self.B = np.zeros((self.m,self.K-1)) #we should probably think of a better guess 
      self.ls_origin_to_current_pos()
   #}
   
   def value(self):
   #{
      #optimziers are set up for minimization, so we will be returning the -log likelihood
      
      #our objective function (without sign adjustment) is:
      #
      # Obj =  term1 + term2
      #
      #  term1 =  -\sum_l^N ln(1 + \sum_{a=1}^{K-1} EXP[(XB)_{la}])
      #  term2 =  +\sum_{i=1}^{K-1} (XB)_{(i)i}  --- where (i) indicates rows for category i obs
      #
      
      #common
      XB = np.dot(self.X,self.B)
      
      #term1 specifics
      exp_XB = np.sum(np.exp(XB),axis=1) #element-wise exponential, sum over rows (categories)
      term1 = -1.0* np.sum(np.log(exp_XB + 1.0)) #element-wise natural log 
      
      #term2 specifics
      term2 = 0.0
      row_off = 0
      for i in range(self.K-1):
         if self.obs_cat[i] > 0:
            term2 = term2 + np.sum(XB[row_off:row_off+self.obs_cat[i],i])
            row_off = row_off + self.obs_cat[i]
      
      #combine
      obj = -1.0*(term1 + term2) #make it minimization
      return obj
   #}
   
   def gradient(self):
   #{
      #optimziers are set up for minimization, so we will be returning the -log likelihood grad
      
      #our gradient (without sign adjustment) is:
      #
      # \frac{\partial Obj}{\partial B_{cd}} =  term1_{cd} + term2_{cd}
      #
      #  term1_{cd} =  -\sum_l^N  [  (X_{lc} EXP[(XB)_{ld}])  /  (1 + \sum_{a=1}^{K-1}  EXP[(XB)_{la}]) ] 
      #  term2_{cd} =  +\sum_l^N I(l in d) X_{lc}  
      #
      
      #term1
      XB = np.dot(self.X,self.B)
      exp_XB = np.exp(XB)
      #crank out the denominator of term1
      denom = np.divide(np.ones(self.N),np.sum(exp_XB,axis=1) + 1.0) #element-wise exponential, sum over rows (categories)
      #divide is element-wise divide
      temp = np.multiply(exp_XB,denom[:,np.newaxis]) #apply denominator. scale rows of exp_XB
      term1 = -1.0*np.dot(self.X.T,temp)
      
      #term2
      term2 = np.zeros((self.m,self.K-1))
      row_off = 0
      for d in range(self.K-1):
         if self.obs_cat[d] > 0:
            for c in range(self.m):
               term2[c,d] = np.sum(self.X[row_off:row_off+self.obs_cat[d],c])
            row_off = row_off + self.obs_cat[d]
      
      #combine
      grad = -1.0*(term1 +term2)
      #make into a column vector
      grad = np.reshape(grad,grad.size,order='F') #column major 
      
      #########
      #DEBUG
      do_debug = False
      if do_debug:
         fd_grad = self.finite_difference_gradient()
         error = grad - fd_grad
         rms_err = math.sqrt(np.dot(error,error)/float(len(error)))
         print "rms error analytic vs finite different gradient ="+"{:.12f}".format(rms_err)
      #########
      
      return grad 
   
   #}
   
   def finite_difference_gradient(self,step_size=1.0E-4):
   #{
      
      #store our current position
      B_store = np.array(self.B)

      disp_len = self.B.size
      grad = np.zeros(disp_len)
      for i in range(disp_len):
         disp = np.zeros(disp_len)
         #forward step
         disp[i] = step_size
         self.update(disp)
         v_forward = self.value()
         #return to fd origin
         self.B = np.array(B_store)
         #back step
         disp[i] = -1.0*step_size
         self.update(disp)
         v_back = self.value()
         #return to fd origin
         self.B = np.array(B_store)
         #compute gradient for this parameter 
         grad[i] = (v_forward - v_back)/(2.0*step_size)
      return grad
   #}
   
   def update(self,disp):
   #{
      disp_reshaped = np.reshape(disp,(self.m,self.K-1),order='F') #column major. consistent with grad 
      self.B = self.B + disp_reshaped
      return 
   #}
   
   def ls_origin_to_current_pos(self):
   #{
      self.B_origin = np.array(self.B)
      return 
   #}
   
   def move_to_ls_origin(self):
   #{
      self.B = np.array(self.B_origin)
      return
   #}
   
   def train(self,tol=1.0E-7,print_iter=False):
   #{
      self.tolerance = tol
      optimizer = l_bfgs(self,40,0,0.05,0)
      max_iter = 5000
      converged = False
      cur_iter = 0
      print 'beginning minimization of negative log likelihood for logistic regression'
      for i in range(max_iter):
         cur_iter = cur_iter + 1
         converged = optimizer.next_step() 
         if (converged):
            break
         if (print_iter):
            print "  "+str(cur_iter)+"  "+"{:.12f}".format(optimizer.value)+"  "+str(optimizer.error)+"  "+optimizer.comment

      if (converged):
         print "  "+str(cur_iter)+"  "+"{:.12f}".format(optimizer.value)+"  "+str(optimizer.error)+"  Optimization Converged"
      else:
         print "  "+str(cur_iter)+"  "+"{:.12f}".format(optimizer.value)+"  "+str(optimizer.error)+"  Optimization Failed"
      return converged
   #}
   
   def predict(self,X_predict):
   #{ 
      #for each observation (row) x_i we have:
      #log(P(G=1|x_i)/P(G=K|x_i)) = x_i B.col(1) 
      #(where row x_i has a 1 added at the end to account for the intercept)
      #
      #if all log ratios are negative, then we classify to group K
      
      X_aug = np.column_stack((X_predict,np.ones(X_predict.shape[0]))) #add columns of 1's for intercepts
      log_ratios = np.dot(X_aug,self.B)
      y_predict = np.zeros(X_aug.shape[0],dtype=np.int)
      y_predict.fill(self.K-1) #only update if the answer isn't the class in the denom of the log ratio
      for i in range(log_ratios.shape[0]):
         max_i = np.argmax(log_ratios[i,:])
         if log_ratios[i,max_i]>0.0:
            y_predict[i] = max_i

      return y_predict
   #}
   
#}
