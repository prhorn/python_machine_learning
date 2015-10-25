import numpy as np
import math
import sys
import matplotlib.pyplot as plt

#Least Angle Regression
#
#Notation will be close to:
#  B. Efron, T. Hastie, I. Johnstone, R. Tibshirani,
#  The Annals of Statistics, 32, 407 (2004).

class LARS:
#{
   #Quantities from centering and norming input data
   #needed to reverse these transformations
   #
   #y_mean
   #x_means
   #x_norms
   
   #centered and norm'd data
   #X -- covariates, nxm
   #y -- prediction target, n-long
   #n -- rows of X and y
   #m -- cols of X

   #beta             -- our coefs, m-long
   #mu_hat           -- our current prediction, n-long
   #active_indices   -- list of indices of active columns of X
   
   #mode -- 0, least angle regression
   #     -- 1, lasso
   #     -- 2, stagewise NYI
   
   def __init__(self,X_,y_,mode_=0):
   #{
       
      self.mode = mode_
      if not ((self.mode == 0) or (self.mode == 1) or (self.mode == 2)):
         print "mode not recognized in LARS"
         sys.exit(1)
      if self.mode == 2:
         print "stagewise NYI"
         sys.exit(1)

      self.X = np.array(X_)
      self.y = np.reshape(y_,len(y_))
      
      self.n = self.X.shape[0] #rows of X, number of observations
      self.m = self.X.shape[1] #cols of X, number of covariates
      if not (self.y.size == self.n):
         print "number of observations (rows) in X and y did not match in LARS constructor"
         sys.exit(1)
      
      #centering and norming
      self.x_means = np.mean(self.X, axis=0)
      for i in range(self.m):
         self.X[:,i] =  self.X[:,i] - self.x_means[i]
      #
      self.x_norms = np.zeros(self.m)
      for i in range(self.m):
         self.x_norms[i] = math.sqrt(np.dot(self.X[:,i],self.X[:,i])) 
         self.X[:,i] =  (1.0/self.x_norms[i])*self.X[:,i]
      #
      self.y_mean = np.mean(self.y) 
      self.y = self.y - self.y_mean
      
      #initialize iterated quantities
      self.beta = np.zeros(self.m)
      self.mu_hat = np.zeros(self.n)
      self.active_indices = []

   #}
   
   def next_step(self):
   #{
      debug = False

      c = np.dot(self.X.T,self.y - self.mu_hat)
      C = max(c.min(),c.max(), key=abs)
      
      if debug:
         print 'correlations'
         print c
         print 'this is big C ',C

      if len(self.active_indices)==0:
         if abs(c.max()) > abs(c.min()):
            self.active_indices.append(c.argmax())
         else:
            self.active_indices.append(c.argmin())

      s_active = [] 
      for i in self.active_indices:
         if c[i] < 0.0:
            s_active.append(-1.0)
         else:
            s_active.append(1.0)
      
      if debug:
         print 'this is s_active'
         print s_active

      X_active = self.X[:,self.active_indices]
      for i in range(X_active.shape[1]):
         if s_active[i] < 0.0:
            X_active[:,i] = -1.0*X_active[:,i]
         #else we mult by 1.0...
      
      G_active = np.dot(X_active.T,X_active)
      G_active_inv = np.linalg.pinv(G_active) 
      #DEBUG
      if not (np.allclose(np.identity(G_active.shape[0]), np.dot(G_active,G_active_inv))):
         print 'pseudoinverse of G was poor'
      A_active = math.sqrt(np.sum(G_active_inv)) 
      w_active = A_active*np.dot(G_active_inv,np.ones(G_active.shape[0]))
      u_active = np.dot(X_active,w_active)
      a = np.dot(self.X.T,u_active)
      gamma = 1.0E12
      updated_gamma = False
      next_index = -1
      
      for i in range(self.m):
         if not i in self.active_indices:
            plus = (C + c[i])/(A_active + a[i])
            minus = (C - c[i])/(A_active - a[i])
            if (plus>0.0) and (plus<gamma):
               if debug:
                  print 'we chose plus, gamma = ',plus,' index is ',i
               gamma = plus
               next_index = i
               updated_gamma = True
            if (minus>0.0) and (minus<gamma):
               if debug:
                  print 'we chose minus, gamma = ',minus,' index is ',i
               gamma = minus
               next_index = i
               updated_gamma = True
      if not updated_gamma:
         if not (len(self.active_indices)==self.m):
            print 'we seem to be unable to progress though we have not included all parameters'
         gamma = C/A_active
      
      if debug:
         print 'this is our projected next max correlation ',(C-gamma*A_active)
      
      twiddle_index = -1 #next index is index to add. twiddle index is the one to remove
      if (self.mode == 1):
         #LASSO modification
         gamma_twiddle = gamma + 10.0 #we only care if gamma_twiddle is less than gamma
         for j in range(len(self.active_indices)):
            gamma_j = -1.0*self.beta[self.active_indices[j]]/(s_active[j]*w_active[j])
            if (gamma_j>0.0) and (gamma_j<gamma_twiddle):
               gamma_twiddle = gamma_j
               twiddle_index = j
         if (gamma_twiddle < gamma):
            gamma = gamma_twiddle
            next_index = -1 #we aren't adding indices. we are subtracting this iter
         else:
            twiddle_index = -1 #make sure it is -1 if LASSO mod is not relevant

      #update coefficients and mu
      self.mu_hat = self.mu_hat + gamma*u_active 
      for j in range(len(self.active_indices)):
         self.beta[self.active_indices[j]] = self.beta[self.active_indices[j]] + gamma*w_active[j]*s_active[j]
      
      if debug:
         print 'coefficients (for centered data)'
         print self.beta

      if (next_index >= 0):
         if debug:
            print 'this is the next index ',next_index
         self.active_indices.append(next_index)
         return True,C    
      elif (twiddle_index >= 0): #Lasso modification came into play 
         temp = self.active_indices.pop(twiddle_index)
         print 'removing active index ',temp 
         return True,C    
      
      if debug:
         c = np.dot(self.X.T,self.y - self.mu_hat)
         print 'these are our final correlations '
         print c
      
      return False,C
   #}
   
   def train(self):
   #{
      iteration = 0
      keep_going = True
      
      C_list = []
      beta_list = []

      while keep_going:
         beta_list.append(np.array(self.beta))
         keep_going,C = self.next_step()
         C_list.append(C)  
      #get last C to confirm zero
      beta_list.append(np.array(self.beta))
      keep_going,C = self.next_step()
      C_list.append(C)
      
      #make them members to make analysis easier for now
      self.max_corr_vs_iter = C_list
      self.coefs_vs_iter = [[] for i in range(self.m)]
      for b_i in beta_list:
         for i in range(self.m):
            self.coefs_vs_iter[i].append(b_i[i])
      
      self.make_plots()
   #}
   
   def make_plots(self):
   #{
      fig = plt.figure()
      ax = fig.add_subplot(111)
      iterations = range(len(self.max_corr_vs_iter))
      ax.scatter(iterations,self.max_corr_vs_iter,s=10,c='b',marker="s",label="Max Absolute Correlation")
      plt.legend(loc='best');
      plt.show()
   
      fig = plt.figure()
      ax = fig.add_subplot(111)
      for i in range(self.m):
         ax.plot(iterations,self.coefs_vs_iter[i],'xb-',label="Coefficient "+str(i))
      plt.legend(loc='upper left');
      plt.show()
   
   #}
#}
