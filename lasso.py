import numpy as np
import scipy.linalg
import sys
import math
import os
sys.path.append(os.getcwd()+'/nonlinear_solvers/')
from l_bfgs import l_bfgs

class lasso: #NUMERICS ARE HORRIBLE. use LARS instead
#{
   #
   #  Fit linear (y = X\beta + \epsilon) 
   #  model with L1 regularization (LASSO)
   #
   #  lambd          --    the regularization parameter
   #  add_constant   --    bool determining whether to optimize
   #                       a constant term in additon to our passed X.
   #                       We do not include the constant in the
   #                       penalty term
   #  is_weighted    --    indicate if we are weighted least squares
   #  weights_diag   --    indicates whether weight matrix is diag
   #  X              --    passed features for the model
   #  y              --    passed quantity to be predicted by the model
   #  beta           --    our fitted coefficients
   #  beta_0         --    our intercept (if we have one)
   #  W              --    weight matrix (if we have one)
   #                       if weights_diag, then this is a vector
   #                       if a matrix, then it is assumed symmetric

   def __init__(self,X_,y_,add_constant_=True,W_=None,is_weighted_=False,weights_diag_=True):
   #{
      self.tolerance = 1.0E-5
      self.X = np.array(X_)
      self.y = np.reshape(y_,len(y_))
      self.add_constant = add_constant_
      self.W = np.array(W_)
      self.is_weighted = is_weighted_
      self.weights_diag = weights_diag_

      N = self.X.shape[0] #rows of X, number of observations
      if not (self.y.shape[0] == N):
         print "number of observations (rows) in X and y did not match in lasso constructor"
         sys.exit(1)
      if self.is_weighted:
         if self.weights_diag:
            if not (self.W.shape[0] == N):
               print "number of observations (rows) in X and length of passed weights vector do not match in lasso constructor"
               sys.exit(1)
         else:
            if not ((self.W.shape[0] == N) and (self.W.shape[1] == N)):
               print "number of observations (rows) in X and the number of rows and cols of the weights matrix do not match in lasso constructor"
               sys.exit(1)
         
   #}
   
   def value(self):
   #{
      #our objective function is:
      #  obj = e^T W e + \lambda  \sum_i |\beta_i| (\beta_0 not in sum)
      #  e = y - X\beta - \beta_0
      #if we don't have weights, W = I
      #if we don't have an intercept, \beta_0 = 0 
      
      #get a common intermediate out of the way
      #print 'shape of y ',self.y.shape
      #print 'shape of X ',self.X.shape
      #print 'shape of beta ',self.beta.shape
      e = self.y - np.dot(self.X,self.beta)
      if self.add_constant:
         e = e - self.beta_0
      #print 'shape of e ',e.shape 
      #take care of weight cases
      if self.is_weighted:
         if self.weights_diag:
            obj = np.dot(e,np.dot(np.diag(self.W),e))
         else:
            obj = np.dot(e,np.dot(self.W,e))
      else:
         obj = np.dot(e,e)
      
      obj = obj

      #add the regularization term
      penalty = 0.0
      for i in range(self.beta.size):
         penalty = penalty + abs(self.beta[i])
      penalty = self.lambd * penalty
      obj = obj + penalty
      
      return obj
   #}
    
   def gradient(self):
   #{ 
      #in the following:
      # aug_beta is a coluumn vector with beta stacked on top of beta_0
      # aug_X is X with a column of ones added to the end
      # 
      #the gradient is the sum of two terms:
      #
      #lsq is length of beta + 1 (if constant) long
      #  lsq = 2.0*(aug_X^T W aug_X aug_beta - aug_X^T W y)
      #
      #pen is length beta long and does not involve the consant
      #
      #  pen_j = \lambda beta_j/abs(beta_j) = lambda * sign(beta_j) 
      
      if self.add_constant:
         aug_X = np.column_stack((self.X,np.ones(self.X.shape[0])))
         lsq = np.dot(aug_X,np.concatenate((self.beta,np.array([self.beta_0])))) - self.y
         if self.is_weighted:
            if self.weights_diag:
               lsq = np.dot(np.transpose(aug_X),np.dot(np.diag(self.W),lsq))
            else:
               lsq = np.dot(np.transpose(aug_X),np.dot(self.W,lsq))
         else:
            lsq = np.dot(np.transpose(aug_X),lsq)
      else:
         lsq = np.dot(self.X,self.beta) - self.y
         if self.is_weighted:
            if self.weights_diag:
               lsq = np.dot(np.transpose(self.X),np.dot(np.diag(self.W),lsq))
            else:
               lsq = np.dot(np.transpose(self.X),np.dot(self.W,lsq))
         else:
            lsq = np.dot(np.transpose(self.X),lsq)
      lsq = lsq * 2.0
      
      pen = np.zeros(len(self.beta),dtype=np.float)
      for i in range(len(pen)):
         if self.beta[i] > 0.0:
            pen[i] = self.lambd
         else:
            pen[i] = -1.0*self.lambd
      
      if self.add_constant:
         grad = lsq
         grad[:-1] = grad[:-1] + pen
      else:
         grad = lsq + pen
      
      #print 'the shape of grad ',grad.shape
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
      beta_fd_store = np.array(self.beta)
      if self.add_constant:
         beta_0_fd_store = self.beta_0
      
      disp_len = self.beta.size
      if self.add_constant:
         disp_len = disp_len + 1
      
      grad = np.zeros(disp_len)
      for i in range(disp_len):
         disp = np.zeros(disp_len)
         #forward step
         disp[i] = step_size
         self.update(disp)
         v_forward = self.value()
         #return to fd origin
         self.beta = np.array(beta_fd_store)
         if self.add_constant:
            self.beta_0 = beta_0_fd_store
         #back step
         disp[i] = -1.0*step_size
         self.update(disp)
         v_back = self.value()
         #return to fd origin
         self.beta = np.array(beta_fd_store)
         if self.add_constant:
            self.beta_0 = beta_0_fd_store
         #compute gradient for this parameter 
         grad[i] = (v_forward - v_back)/(2.0*step_size)
      return grad
   #}

   def obtain_guess(self):
   #{
      #lets try guessing ridge regression with the same regularization
      if self.add_constant:
         self.beta_0 = np.sum(self.y)/float(len(self.y))
         y_centered = self.y - self.beta_0
      else:
         y_centered = np.array(self.y) #lazy
      #metric = Xt W X + \lambda I
      if self.is_weighted:
         if self.weights_diag:
            metric = np.dot(np.transpose(self.X),np.dot(np.diag(self.W),self.X)) 
         else:
            metric = np.dot(np.transpose(self.X),np.dot(self.W,self.X)) 
      else:
         metric = np.dot(np.transpose(self.X),self.X)
      metric = metric + np.diag(np.ones(self.X.shape[1])*self.lambd) 
      U,s,V = np.linalg.svd(metric) #NB svals are decreasing
      tol = 1.0E-8
      n_indep = s.size
      for i in s:
         if (i < tol):
            n_indep -= 1

      if n_indep > 0:
         metric_inv = np.dot(U[:,:n_indep],np.dot(np.diag(s[:n_indep]**-1),np.transpose(U[:,:n_indep])))
      else:
         print 'poorly posed problem in linear_regression'
         sys.exit(1)
      # beta = metric^{-1} Xt W y
      if self.is_weighted:
         if self.weights_diag:
            self.beta = np.dot(metric_inv,(np.dot(np.transpose(self.X),np.dot(np.diag(self.W),y_centered))))
         else:
            self.beta = np.dot(metric_inv,np.dot(np.transpose(self.X),np.dot(self.W,y_centered)))
      else:
         self.beta = np.dot(metric_inv,np.dot(np.transpose(self.X),y_centered))
            
      #print 'PRINTING DIMS IN GUESS'   
      #print 'X shape ',self.X.shape
      #print 'y shape ',self.y.shape
      #print 'weights shape ',self.W.shape
      #print 'beta shape ',self.beta.shape
      #print 'beta type ',type(self.beta)
      #print 'beta[0] ',self.beta[0]
      #print 'beta[0] type ',type(self.beta[0])
      #print 'END PRINTING DIMS IN GUESS'   
   #}
   
   def obtain_guess_for_lambda(self,passed_lambda):
   #{
      self.lambd = passed_lambda
      if (self.lambd<0.0):
         print "regularization parameter, lambda, in LASSO cannot be negative"
         sys.exit(1)
      
      self.obtain_guess()
   #}
   
   def update(self,disp):
   #{
      #ORDER:
      #  beta    
      #  beta_0  
      #print 'shape of disp ',disp.shape
      #print 'shape of beta before update ',self.beta.shape
      if self.add_constant:
         self.beta = self.beta + disp[:-1]
         self.beta_0 = self.beta_0 + disp[-1]
      else:
         self.beta = self.beta + disp
      #print 'shape of beta after update ',self.beta.shape
      
      return 
   #}
   
   def ls_origin_to_current_pos(self):
   #{
      self.beta_o = np.array(self.beta)
      if self.add_constant:
         self.beta_0_o  =  self.beta_0
      return 
   #}
   
   def move_to_ls_origin(self):
   #{
      self.beta = np.array(self.beta_o)
      if self.add_constant:
         self.beta_0  =  self.beta_0_o
      return
   #}

   def train(self,lambda_,reguess=True):
   #{
      self.lambd = lambda_
      if (self.lambd<0.0):
         print "regularization parameter, lambda, in LASSO cannot be negative"
         sys.exit(1)
      
      if reguess:
         self.obtain_guess()

      optimizer = l_bfgs(self,40,0,0.05,0)
      max_iter = 5000
      converged = False
      cur_iter = 0
      print_iter = True
      print 'beginning minimization of LASSO objective function with lambda = '+str(self.lambd)
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
#}
