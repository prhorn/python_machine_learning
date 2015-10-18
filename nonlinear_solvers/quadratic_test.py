import numpy as np
import sys
from l_bfgs import l_bfgs

class quadratic_test:
#{
   #  H is sym SPD
   #obj = sum_i (q_i x_i)  + 0.5 sum_ij (x_i H_ij x_j)
   #grad_k = q_k + sum_j (H_kj xj)
   #hess_kl = H_kl
   
   #x - our variables
   #x_origin - variables at the origin of the line search
   #H - our hessian 
   #q - linear contriubtion to our objective function
   #tolerance - our threshold for convergence

   def __init__(self,size): 
   #{
      if (size<1):
         print 'size must be positive'
         sys.exit(1)
      
      self.tolerance = 1.0E-6
      
      self.q = np.random.rand(size)
      self.x = np.zeros(size) 
      self.x_origin = np.zeros(size) 
      
      #make H
      L = np.random.rand(size,size)
      for r in range(size): 
         for c in range(r+1,size):   
            L[r,c] = 0.0
      self.H = np.dot(L,np.transpose(L)) + np.ones(shape=(size,size),dtype=np.float)
      self.H = 0.01*self.H

      #test H
      evals = np.linalg.eigvalsh(self.H)
      if (min(evals) < 0.0):
         print 'We did not produce a PSD H. Smallest eval: ',min(evals)
         sys.exit(1)
   
   #}
   
   def value(self):
   #{
      #obj = sum_i (q_i x_i)  + 0.5 sum_ij (x_i H_ij x_j)
      obj = np.dot(self.x,self.q) + 0.5*np.dot(self.x,np.dot(self.H,self.x))
      return obj
   #}
   
   def gradient(self):
   #{
      #grad_k = q_k + sum_j (H_kj xj)
      grad = self.q + np.dot(self.H,self.x)
      return grad
   #}

   def update(self,direction):
   #{
      self.x = self.x + direction
   #}
   
   def obtain_guess(self):
   #{
      self.x = np.zeros(self.q.size)
      self.x_origin = self.x
   #}

   def ls_origin_to_current_pos(self):
   #{
      self.x_origin = self.x 
   #}
   
   def move_to_ls_origin(self):
   #{
      self.x = self.x_origin 
   #}
   
   def solve(self,max_iter):
   #{
      self.obtain_guess() 
      optimizer = l_bfgs(self)

      converged = False
      cur_iter = 0
      for i in range(max_iter):
         cur_iter = cur_iter + 1
         converged = optimizer.next_step() 
         if (converged):
            break
         print "  "+str(cur_iter)+"  "+str(optimizer.value)+"  "+str(optimizer.error)+"  "+optimizer.comment

      if (converged):
         print "  "+str(cur_iter)+"  "+str(optimizer.value)+"  "+str(optimizer.error)+"  Optimization Converged"
      else:
         print 'optimization failed to converge'
   #}
#}
