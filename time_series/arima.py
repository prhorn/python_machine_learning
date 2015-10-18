import numpy as np
import scipy.linalg
import sys
import math
import os
sys.path.append(os.getcwd()+'/../nonlinear_solvers/')
from l_bfgs import l_bfgs

class ARIMA:
#{
   #maximum likelihood

   #Basic Parameters
   #  p  -  of AR(p)
   #  d  -  of I(d)
   #  q  -  of MA(q)
   
   #  z  -  training data
   #  w  -  differenced training data with zero mean
   #  mu -  mean of differenced training data

   #optimized parameters (in gradient order)
   #  n_param     -  number of parameters (length of gradient)
   ## e_star      -  past w (p-long) followed by past a (q-long)  
   ##                w_{1-p},...,w_{0},a_{1-q},...,a_{0}
   ##                (only for documentation)
   #  sigma_a_sq  -  variance of a, shocks
   #  phi         -  AR parameters
   #  theta       -  MA parameters
   
   #parameters at LS origin for optimizers that use LS
   #  sigma_a_sq_o
   #  phi_o
   #  theta_o

   #intermediates matrices
   #  Omega       -  (p+q)x(p+q) E[e_star e_star^T]/sigma_a_sq 
   #  F
   #  L_theta
   #  L_phi
   #  D
   #  e_star_hat
   #  a_hat  

   def __init__(self,p_,d_,q_):
   #{
      #basic
      self.p         = p_
      self.d         = d_
      self.q         = q_
      if ((self.p<0) or (self.d<0) or (self.d<0)):
         print 'all ARIMA(p,d,q) model parameters must be positive'
         sys.exit(1)
      
      self.n_param   = 1 + self.p + self.q
      self.tolerance = 1.0E-7

      #optimized parameters
      self.sigma_a_sq= 1.0
      self.phi       = np.empty(0)
      self.theta     = np.empty(0)
      
      #optimized parameters_o
      self.ls_origin_to_current_pos()
      
      #intermediates
      self.Omega     = np.empty((0,0))
      self.F         = np.empty((0,0))
      self.L_theta   = np.empty((0,0))
      self.L_phi     = np.empty((0,0))
      self.D         = np.empty((0,0))
      self.e_star_hat= np.empty(0)
      self.a_hat     = np.empty(0)

   #}
   
   def obtain_guess(self,guess_phi,guess_theta,guess_sigma_a_sq):
   #{
      if not (len(guess_phi) == self.p):
         print 'guess_phi must be p-long'
         sys.exit(1)
      if not (len(guess_theta) == self.q):
         print 'guess_theta must be q-long'
         sys.exit(1)
      
      self.phi = np.array(guess_phi)
      self.theta = np.array(guess_theta)
      self.sigma_a_sq = float(guess_sigma_a_sq)

      self.update_intermediates()
      return
   #}
   
   def update_intermediates(self):
   #{
      self.compute_Omega()
      self.compute_FLL()
      self.compute_D()
      self.compute_e_star_hat()
      self.compute_a_hat()  
      return
   #}

   def compute_Omega(self):
   #{
      #              p                    q
      #Omega = p [   Gamma/sigma_a_sq  |  C^T   ]
      #        q [   C                 |  I     ]
      
      #get trivial cases out of the way
      if not ((self.p + self.q) > 0):
         #nothing to do
         self.Omega = np.empty((0,0))
         return
      elif (self.p == 0):
         #easy
         self.Omega = np.identity(self.q)
         return
      #else: Omega is more involved and will be returned at end of the function

      #we need psi, the coefficients for the infinite MA process
      #for both Gamma and C
      #Gamma needs psi from 0 to q
      #C needs psi from 0 to q-1
      if (self.q == 0):
         psi = [1.0]
      else:
         psi = [1.0]*(self.q+1)
         for j in range(1,self.q+1): #zero is covered. compute 1,2,...,q
            #compute psi_j
            #Box page 78
            #psi_j = \sum_{i=1}^{p} phi_{i} psi_{j-i} - theta_{j}
            #  psi_0 = 1.0
            #  psi_{<0} = 0
            #  theta_j = 0 for j>q ## not a concern for these j
            psi[j] = -1.0*self.theta[j-1] #theta_1 is at array pos 0
            for i in range(1,self.p+1):
               if ((j-i)> -1):
                  psi[j] = psi[j] + self.phi[i-1] * psi[j-i] #phi_1 is a array pos 0

      #make pxp Gamma
      if (self.p>0):
         #Gamma_ij = gamma_{abs(i-j)}
         #we need gamma from 0 to p-1
         #but we will compute 0 to p
         #gamma_k is the autocovarance at lag k
         #  (Ljung and Box)
         #  A * gamma = v (linear equation)
         #  v_j = psi_{1-j} - \sum_{i=1}^{q} theta_{i} psi_{i+1-j}      (j = 1,...,p+1) 
         #     NB: psi_{<0} = 0
         v = np.zeros(self.p+1)
         for j in range(1,self.p+2):
            #v_j = v[j-1]
            if (j == 1):
               v[j-1] = v[j-1] + psi[0]
            for i in range(1,self.q+1):
               #theta_i = theta[i-1]
               temp = i+1-j
               if (temp > -1):
                  v[j-1] = v[j-1] - self.theta[i-1] * psi[temp]
         #now construct A
         A = np.identity(self.p+1)
         for i in range(self.p+1):
            for j in range(self.p+1):
               if (i>j): 
                  A[i,j] = A[i,j] - self.phi[i-j-1]
               if ((j>0) and ((i+j)<=self.p)):
                  A[i,j] = A[i,j] - self.phi[j+i-1]
         gamma = np.linalg.solve(A,v)
         #fill in Gamma with gamma values
         Gamma = np.zeros((self.p,self.p))
         for i in range(self.p):
            for j in range(self.p):
               Gamma[i,j] = gamma[abs(i-j)]
      
      #make qxp C
      if ((self.p>0) and (self.q>0)):
         #C = E[a_star w_star^t]/sigma_a_sq
         #
         #with row indices going from 1 to q
         #and  col indices going from 1 to p
         #we have:
         #C_ij = E[a_{i-q} w_{j-p}^T]/sigma_a_sq
         #     = psi_{j-i-p+q} = psi_{(j-p) - (i-q)}
         #     for (j-p) - (i-q) >= 0
         #which means we need psi from 0 to q-1 (q values)
         #with psi_0 = 1.0
         #we have already constructed the part of psi that we need above
         #now construct C
         C = np.zeros((self.q,self.p),dtype=np.float)
         for i in range(1,self.q+1):
            for j in range(1,self.p+1):
               temp = j-i-self.p+self.q
               if (temp > -1):
                  C[i-1,j-1] = psi[temp]
               
      #form Omega from the pieces
      self.Omega = np.zeros((self.p+self.q,self.p+self.q))
      #if p is zero or if p and q are both zero, then we have already returned
      #at least p is nonzero
      self.Omega[:self.p,:self.p] = Gamma/self.sigma_a_sq
      if (self.q>0):
         self.Omega[self.p:,self.p:] = np.identity(self.q)
         self.Omega[self.p:,:self.p] = C
         self.Omega[:self.p,self.p:] = np.transpose(C)
      
      return
   #}
   
   def compute_FLL(self):
   #{
      #compute F, L_theta, and L_phi members
      #based on members theta and phi
      
      if (len(self.w)<self.p) or (len(self.w)<self.q):
         print 'not dealing with this silly case of p or q greater than training points'
         sys.exit(1)

      #F
      self.F = np.zeros((len(self.w),self.p+self.q))
      if ((self.p + self.q) > 0):
         if (self.p>0):
            A_p = np.zeros((self.p,self.p))
            for i in range(self.p):
               for j in range(i,self.p): #upper triangle
                  A_p[i,j] = self.phi[self.p-1-(j-i)]
            self.F[:self.p,:self.p] = A_p
         if (self.q>0):
            B_q = np.zeros((self.q,self.q))
            for i in range(self.q):
               for j in range(i,self.q):
                  B_q[i,j] = -1.0*self.theta[self.q-1-(j-i)]
            self.F[:self.q,self.p:] = B_q
   
      #L_theta
      self.L_theta = np.identity(len(self.w)) #lower triangular
      #-theta_i on the ith subdiagonal
      if (self.q>0):
         for j in range(len(self.w)): #cols
            for i in range(j+1,len(self.w)): #rows
               temp = i-j-1
               if (temp<self.q):
                  self.L_theta[i,j] = -1.0*self.theta[temp]
      
      #L_phi
      self.L_phi = np.identity(len(self.w)) #lower triangular
      if (self.p>0):
         for j in range(len(self.w)): #cols
            for i in range(j+1,len(self.w)): #rows
               temp = i-j-1
               if (temp<self.p):
                  self.L_phi[i,j] = -1.0*self.phi[temp]
         
      return
   #}
   
   def compute_D(self):
   #{
      #D = Omega^{-1} + (L_theta^{-1} F)^T (L_theta^{-1} F)
      #  = Omega^{-1} + X^T X
      #where: L_theta X = F
      if not ((self.p + self.q) > 0):
         #nothing to do
         self.D = np.empty((0,0))
         return
      
      if (self.q == 0):
         X = self.F
      else:
         X = scipy.linalg.solve_triangular(self.L_theta,self.F,lower=True,unit_diagonal=True)
      
      #tiny inverse. should be fine
      self.D = np.linalg.inv(self.Omega) + np.dot(np.transpose(X),X)
      
      return
   #}

   def compute_e_star_hat(self):
   #{
      #e_star_hat are the optimal back-forecased  values 
      if not ((self.p + self.q) > 0):
         #nothing to do
         self.e_star_hat = np.empty(0)
         return
      
      # D e_star_hat = F^T u
      # u_t = a_zero_t + sum_{i=1}^{q} theta_i u_{t+1} 
      #     backwards recursion
      #     for t = 1 through n = len(w) 
      #     u_{>n} = 0
      # a_zero = L_theta^{-1} L_phi w
      
      #get a_zero
      b = np.dot(self.L_phi,self.w)
      if (self.q == 0):
         a_zero = b
      else:
         a_zero = scipy.linalg.solve_triangular(self.L_theta,b,lower=True,unit_diagonal=True)
      
      #get u
      u = np.zeros(len(self.w))
      for j in reversed(range(len(self.w))):
         u[j] = a_zero[j]
         for i in range(self.q):
            if ( (j+i+1) < len(self.w) ):
               u[j] = u[j] + self.theta[i] * u[j+i+1]
      
      #get e_star_hat
      b = np.dot(np.transpose(self.F),u)
      self.e_star_hat = np.linalg.solve(self.D,b)
      return
   #}
   
   def compute_a_hat(self):
   #{
      #L_theta a_hat = L_phi w - F e_star_hat
      
      #L_phi term of RHS
      if (self.p>0):
         b = np.dot(self.L_phi,self.w)
      else:
         b = np.array(self.w)
      
      #F term of RHS
      if ((self.p + self.q)>0):
         b = b - np.dot(self.F,self.e_star_hat)
      
      #solve linear equation if L_theta isn't eye
      if (self.q>0):
         self.a_hat = scipy.linalg.solve_triangular(self.L_theta,b,lower=True,unit_diagonal=True)
      else:
         self.a_hat = np.array(b)
      
      return
   #}
   
   def value(self):
   #{
      #value of the likelihood function
      S = np.dot(self.a_hat,self.a_hat) 
      if ((self.p + self.q)>0):
         S = S + np.dot(self.e_star_hat,np.dot(np.linalg.inv(self.Omega),self.e_star_hat))
      
      #omega_det = 1.0
      #d_det = 1.0
      #if ((self.p + self.q)>0):
      #   omega_det = np.linalg.det(self.Omega)
      #   #print 'this is Omega'
      #   #print self.Omega
      #   #print 'this is the determinant of Omega ',omega_det
      #   omega_det = 1.0/math.sqrt(omega_det)
      #   d_det = np.linalg.det(self.D)
      #   d_det = 1.0/math.sqrt(d_det)
      #
      #f_n = float(len(self.w)) 
      #
      #temp_norm = math.pow(2.0*math.pi*self.sigma_a_sq,-1.0*f_n/2.0)
      #temp_exp  = math.exp(-0.5*S/self.sigma_a_sq)
      #print 'S ',S
      #print 'temp_norm ',temp_norm
      #print 'temp_exp ',temp_exp
      #print 'omega_det ',omega_det
      #print 'd_det ',d_det
      #obj =  temp_norm*omega_det*d_det*temp_exp
      
      #print 'Omega '
      #print self.Omega
      #print 'F '
      #print self.F
      #print 'L_theta '
      #print self.L_theta
      #print 'L_phi '
      #print self.L_phi
      #print 'D '
      #print self.D
      #print 'e_star_hat '
      #print self.e_star_hat
      #print 'a_hat'
      #print self.a_hat

      obj = S
      #obj = obj * -1.0 #algo set up for minimization
      return obj
   #}
   
   def gradient(self):
   #{
      #currently implemented with finite difference only
      step_size = 1.0E-4
      
      #store our current position
      sigma_a_sq_fd_store = self.sigma_a_sq
      phi_fd_store  =  np.array(self.phi)
      theta_fd_store = np.array(self.theta)
      
      grad = np.zeros(self.n_param) 
      for i in range(self.n_param):
         disp = np.zeros(self.n_param)
         #forward step
         disp[i] = step_size
         self.update(disp)
         v_forward = self.value()
         #return to fd origin
         self.sigma_a_sq = sigma_a_sq_fd_store
         self.phi  =  np.array(phi_fd_store)
         self.theta = np.array(theta_fd_store)
         #back step
         disp[i] = -1.0*step_size
         self.update(disp)
         v_back = self.value()
         #return to fd origin
         self.sigma_a_sq = sigma_a_sq_fd_store
         self.phi  =  np.array(phi_fd_store)
         self.theta = np.array(theta_fd_store)
         #compute gradient for this parameter 
         #print 'v_forward ',v_forward
         #print 'v_back ',v_back
         grad[i] = (v_forward - v_back)/(2.0*step_size)
      #phi,theta,sigma are at the fd origin
      #return our intermediates to the fd origin
      self.update_intermediates()
      
      print 'gradient'
      print grad

      return grad
   #}
   
   def update(self,disp):
   #{
      #ORDER:
      #  sigma_a_sq  -  variance of a, shocks
      #  phi         -  AR parameters
      #  theta       -  MA parameters
      
      print 'this is our displacement'
      print disp
      self.sigma_a_sq = self.sigma_a_sq + disp[0]
      for i in range(self.p):
         self.phi[i] = self.phi[i] + disp[i+1] 
      for i in range(self.q):
         self.theta[i] = self.theta[i] + disp[i+1+self.p] 
      self.update_intermediates()
      return 
   #}
   
   def ls_origin_to_current_pos(self):
   #{
      self.sigma_a_sq_o = self.sigma_a_sq
      self.phi_o  =  np.array(self.phi)
      self.theta_o = np.array(self.theta)
      
      return 
   #}
   
   def move_to_ls_origin(self):
   #{
      self.sigma_a_sq = self.sigma_a_sq_o
      self.phi  =  np.array(self.phi_o)
      self.theta = np.array(self.theta_o)
      
      return
   #}
   
   def train(self,z_,guess_phi,guess_theta,guess_sigma_a_sq):
   #{
   #  z  -  training data
   #  w  -  differenced training data with zero mean
   #  mu -  mean of differenced training data
      self.z = z_
      self.w = np.array(self.z)
      for i in range(self.d):
         differenced = np.zeros(len(self.w)-1)
         for j in range(len(differenced)):
            differenced[j] = self.w[j+1] - self.w[j]
         self.w = np.array(differenced)
      self.mu = float(np.sum(self.w))/float(len(self.w))
      self.w = self.w - self.mu
      
      self.obtain_guess(guess_phi,guess_theta,guess_sigma_a_sq)
      optimizer = l_bfgs(self)
      
      self.value()
      max_iter = 100
      converged = False
      cur_iter = 0
      print 'beginning minimization of negative of likelihood function'
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

      return converged
   #}
   
   
#}
