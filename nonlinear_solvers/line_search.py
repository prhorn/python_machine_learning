import numpy as np
import sys
import math

#\brief line search with 'Guaranteed Sufficient Decrease'
#   
#   Source:
#      More, J.J. and Thuente D.J. Line Search Algorithms with Guaranteed Sufficient Decrease.
#      ACM Transations on Mathematical Software, Vol 20 No 3 1994 p286-307
#   Notes:
#      I attempt to adhere to the notation in the paper.
#      An unfortunate reality is that the paper assumes exact arithmetic,
#      but this leads to issues when dots of gradients and search directions become
#      +-1E-13. It was thus necessary to add some checks for effecitly 0-sized 
#      search spaces that abort the line search. 
#
###########
#     vecs and ivecs will be np.arrays. double = float
#
#      /*
#         Data about the univariate function phi(alpha) = f(x + alpha*p) 
#         phi is the function for which the wolfe conditions are assessed for convergence
#      */
#      arma::vec phi_prime;       /**< the series of gradients generated along the line. 
#                                       phi_prime(i) = phi_prime(alpha_t(i)) 
#                                 */
#      arma::vec phi;             /**< the series of objective function values generated along the line. 
#                                       phi(i) = phi(alpha_t(i)) 
#                                 */
#      arma::vec alpha_t;         /**< the series of trial step lengths generated. first entry is always 0.0 */ 
#      //Data about I, the interval, at each iter
#      arma::ivec alpha_l;        /**< lower interval bound INDEX. 
#                                       The alpha value for the lower bound is thus alpha_t(alpha_l(i))
#                                       the phi value for the lower bound index is phi(alpha_l(i)) ect
#                                 */         
#      arma::ivec alpha_u;        /**< upper interval bound INDEX. entry is -1 until alpha_u is updated
#                                       from its initial value of +infinity to some value between alpha_min
#                                       and alpha_max
#                                 */
#      /*
#         Data about the auxilary function psi(alpha) = phi(alpha) - phi(0) - mu*alpha*phi_prime(alpha)
#         psi_prime(alpha) = phi_prime(alpha) - mu*phi_prime(0)
#      */
#      arma::vec psi;             /**< the value of the auxilary function at points along the line
#                                      psi(i) = psi(alpha_t(i))                                        
#                                 */
#      arma::vec psi_prime;       /**< the gradient of the auxilary function at points along the line
#                                      psi_prime(i) = psi_prime(alpha_t(i))                                        
#                                 */
#      
#      //Parameters 
#      double alpha_init;         /**< initial guess for line search. should be 1.0 for quasi-newton
#                                      alpha_min<alpha_init<alpha_max
#                                 */
#      double alpha_min;          /**< the minimum alpha that we will allow to be generated. 
#                                      0<= alpha_min
#                                      In practice we will make alpha_min small but nonzero so that we always
#                                      move and the higher level algorithm calling the line search doesn't just
#                                      pick the same direction over and over
#                                 */
#      double alpha_max;          /**< the maximum alpha that we will allow to be generated 
#                                      alpha_max will be determined in the constructor based
#                                      on the type of problem. 
#                                          For unconstrained optimization problems that are bounded 
#                                             from below, this is just a large default value. 
#                                             
#                                          For constrained optimizations this is problem specific and needs to be passed.
#                                             TODO
#
#                                          For unbounded problems M&T suggest:
#                                             alpha_max = (1/mu)*(phi(0)-phi_min)/(-phi_prime(0))
#                                             phi_min = user specified lower bound on phi
#                                             phi_min < phi(0)
#                                             TODO
#
#                                      alpha_min <= alpha_max
#                                 */
#      
#      double mu;                 /**< parameter for the wolfe sufficient decrease condition: 
#                                      phi(alpha) <= phi(0) + mu*alpha*phi_prime(0)  
#                                      mu <= 0.5 for newton and quasi-newton
#                                      
#                                      Nocedal recommends 1E-4 (loose) for quasi-newton
#                                      
#                                 */
#      double eta;                /**< parameter for the wolfe curvature condition:
#                                      fabs(phi_prime(alpha)) <= eta*fabs(phi_prime(0))                                     
#                                       mu <= eta
#
#                                       Nocedal recommends 0.9 (loose) for quasi-newton
#                                 */
#     
#      double delta_max;          /**<  
#                                       factor by which we walk toward alpha_max
#                                       used to update trial alphas until alpha_u changes from its initial value of -1
#                                       
#                                       alpha_t(i+1) in [min(delta_max*alpha_t(i),alpha_max),alpha_max]
#
#                                       alpha_t(i+1) = min(alpha_t(i) + delta_max*(alpha_t(i) - alpha_l(i)),alpha_max) 
#                                       delta_max > 1    
#                                       
#                                       M&T suggest delta_max in [1.1 , 4.0]
#                                 */
#      
#      double delta_min;          /**<
#                                       factor forcing a walk toward alpha_min until alpha_l is updated
#                                       from its initial value of 0
#
#                                       alpha_t(i+1) in [alpha_min,max(delta_min*alpha_t(i),alpha_min)]
#                                       
#                                       the choice of alpha_t(i+1) in this range is determined by interpolation
#
#                                       delta_min < 1
#                                       
#                                       M&T suggest delta_min = 7/12  
#                                       
#                                       NB:
#                                       the constraints on trial values coming from delta_min and delta_max
#                                       will not be active at the same time because an iteration always updates
#                                       alpha_l, alpha_u, or both
#                                 */
#
#      double delta_I;            /**<
#                                       if the length of the interval:
#                                       len I = fabs(alpha_t(alpha_l) - alpha_t(alpha_u))
#                                       does not decrease by a factor of delta_I over
#                                       two (option for this?) cycles
#                                       This only comes into effect after both alpha_l
#                                       and alpha_u have been updated such that alpha_t(alpha_l)
#                                       and alpha_t(alpha_u) are in [alpha_min,alpha_max]
#                                       
#                                       delta_I < 1
#
#                                       M&T suggest delta_I = 0.66  
#                                 */
#      
#      //Internal variables 
#      int iteration;             /**< internal variable for iteration-dependent conditions
#                                      as well as to index alpha, phi, and phi_prime
#                                 */
#      int mode;                  /**< 
#                                      we begin in GSD_MODE_PSI and use psi information to update
#                                      interval bounds and interpolate
#                                      
#                                      when psi(alpha)<=0 && phi_prime(alpha)>0
#                                      we enter (and stay in) GSD_MODE_PHI and use phi information 
#                                      to update interval bounds and interpolate
#
#                                      NB: it is impossible to enter GSD_MODE_PHI without updating
#                                      both the lowe and upper bounds
#                                 */
#      
#      //Printing
#      int iprint;
#      
#      /*
#         Users of this class should only be calling the following:
#            constructor 
#            new_line
#            next_step
#      */
class line_search:
#{
   
   #   Constructor for the line search that initializes parameters based on predefined sets
   #   (calls init_ls_params):
   #   0 = quasi-newton recommendation
   #   default is 0
   def __init__(self,param_pref = 0, _iprint = 0):
   #{
      #we switch between two modes, which we will identify with ints
      self.GSD_MODE_PSI = 0
      self.GSD_MODE_PHI = 1

      self.iprint = _iprint
      self.init_ls_params(param_pref)
      self.check_parameters()
   
      if (self.iprint>0):
         self.print_ls_params()

      self.new_line()

   #}
   
   ##  Constructor that allows one to set all line search parameters. Have fun.
   #def __init__(self,_iprint,_alpha_init,_alpha_min,_alpha_max,_mu,_eta,_delta_min,_delta_max,_delta_I):
   ##{
   #   self.iprint = _iprint
   #   self.alpha_init = _alpha_init
   #   self.alpha_min = _alpha_min
   #   self.alpha_max = _alpha_max
   #   self.mu = _mu
   #   self.eta = _eta
   #   self.delta_min = _delta_min
   #   self.delta_max = _delta_max
   #   self.delta_I = _delta_I

   #   self.check_parameters()

   #   if (self.iprint>0): 
   #      self.print_ls_params()
   #   
   #   self.new_line()

   ##}

   #   Sets line search parameters based on integer input.
   #   Called by line_search constructor.
   #   0 = quasi-newton recommendation
   def init_ls_params(self,param_pref):
   #{
      if (param_pref==0): #quasi newton
         self.alpha_init = 1.0
         self.alpha_min = 1.0E-6
         self.alpha_max = 40.0
         self.mu = 1.0E-4
         self.eta = 0.9
         self.delta_min = 7.0/12.0
         self.delta_max = 2.0
         self.delta_I =  0.66
      elif (param_pref==1): #more strict. newton
         self.alpha_init = 1.0
         self.alpha_min = 1.0E-6
         self.alpha_max = 40.0
         self.mu = 1.0E-2
         self.eta = 0.1
         self.delta_min = 7.0/12.0
         self.delta_max = 2.0
         self.delta_I =  0.66
      elif (param_pref==2): #steepest descent
         #these should probably be tweaked, but who runs SD for anything but testing code?
         self.alpha_init = 1.0
         self.alpha_min = 1.0E-6
         self.alpha_max = 40.0
         self.mu = 1.0E-4
         self.eta = 0.2
         self.delta_min = 7.0/12.0
         self.delta_max = 2.0
         self.delta_I =  0.66
      else: 
         print 'line seach param preference was ',param_pref
         print 'param pref not recognized'
         sys.exit(1) 
      
      self.check_parameters()
   #}
   
   #  Print all of the line search params. Done at the end of the the
   #  constructors if iprint >0
   def print_ls_params(self):
   #{
      print "==============================="
      print "Printing Line Search Parameters"
      print "==============================="
      print " alpha_init =           ",self.alpha_init
      print " alpha_min =            ",self.alpha_min
      print " alpha_max =            ",self.alpha_max
      print " mu (decrease) =        ",self.mu
      print " eta (curvature) =      ",self.eta
      print " delta_max =            ",self.delta_max
      print " delta_min =            ",self.delta_min
      print " delta_I =              ",self.delta_I
      print "==============================="
   #}

   #  Check for inconsistencies in parameters. Called in constructors.
   def check_parameters(self):
   #{
      if not ((0.0<self.mu) and (self.mu<self.eta) and (self.eta<1.0)):
         print "mu = ",self.mu
         print "eta = ",self.eta
         print "must have 0 < mu < eta < 1"
         sys.exit(1)
      
      if not ((self.alpha_min >= 0.0) and (self.alpha_init>self.alpha_min) and (self.alpha_init<self.alpha_max)): 
         print "alpha_min  = ",self.alpha_min  
         print "alpha_init = ",self.alpha_init 
         print "alpha_max  = ",self.alpha_max  
         print "must have 0 <= alpha_min < alpha_init < alpha_max"
         sys.exit(1)

      if not (self.delta_max>1.0):
         print "delta_max must be greater than 1"
         sys.exit(1)
      if not (self.delta_min<1.0): 
         print "delta_min must be less than 1"
         sys.exit(1)
      if not (self.delta_I<1.0):   
         print "delta_I must be less than 1"
         sys.exit(1)
   #}

   #  Print the data we have about the current line to track progress
   #
   def print_ls_status(self):
   #{

      print "====================================================================================================================="
      print "              Printing Line Search Status                                                                            "
      print "====================================================================================================================="
      if (self.mode==self.GSD_MODE_PSI):
         print " mode = PSI"
      elif (self.mode==self.GSD_MODE_PHI):
         print " mode = PHI"
      else:
         print " mode = ?"
      print "====================================================================================================================="
      print " iter        alpha            phi          phi_prime       psi         psi_prime         alpha_l_index  alpha_u_index"
      print "====================================================================================================================="
      for i in range(self.iteration+1):
         print " "+str(i)+"   "+str(self.alpha_t[i])+"   "+str(self.phi[i])+"   "+str(self.phi_prime[i])+"   "+str(self.psi[i])+"   "+str(self.psi_prime[i])+"   "+str(self.alpha_l[i])+"   "+str(self.alpha_u[i])
      print "====================================================================================================================="
   #}

   #  Resets data and internal variables, Allowing one to start a new line search 
   #  with the same line search parameters. 
   def new_line(self):
   #{
      self.iteration = 0
      self.phi = np.array([],dtype=np.float)
      self.phi_prime = np.array([],dtype=np.float)
      self.psi = np.array([],dtype=np.float)
      self.psi_prime = np.array([],dtype=np.float)
      self.alpha_t = np.array([0.0],dtype=np.float)
      self.alpha_l = np.array([0],dtype=np.int) 
      self.alpha_u = np.array([-1],dtype=np.int)
      self.mode  = self.GSD_MODE_PSI
   #}

   #   cur_phi is the current objective function value.
   #   cur_phi_prime is the dot of the current objective function gradient
   #   with the line search direction.
   #   Returns true if the wolfe conditions have been satisfied or the answer
   #   is one of the bounds, alpha_min or alpha_max.
   #   If true is returned, then the alpha at which cur_phi
   #   and cur_phi_prime were evaluated is the final alpha (update unnecessary).
   #   If the line search is not complete, false is returned, and next_alpha 
   #   should be applied.
   #   
   #   returns bool_telling_if_done_ls, next_alpha
   def next_step(self,cur_phi,cur_phi_prime):
   #{
      next_alpha = -777.77 #we will return next_alpha as well as a bool indicating whether we have converged the linesearch 
      #store our new information about phi and psi
      self.phi = np.append(self.phi,cur_phi)
      self.phi_prime = np.append(self.phi_prime,cur_phi_prime)
      cur_psi = self.phi[self.iteration] - self.phi[0] - self.mu*self.phi_prime[0]*self.alpha_t[self.iteration]
      self.psi = np.append(self.psi,cur_psi)
      cur_psi_prime = self.phi_prime[self.iteration] - self.mu*self.phi_prime[0]
      self.psi_prime = np.append(self.psi_prime,cur_psi_prime)
      
      if (self.iprint>0): 
         self.print_ls_status()
      
      #these will be set in the next if/else block if we are going to return false
      next_alpha_l = -7
      next_alpha_u = -7

      if (self.iteration == 0):
         next_alpha = self.alpha_init;
         #in some sense our 0th iter when we store the information
         #at the origin is M&T's -1th iter. we can't update the 
         #interval until we have a non-zero trial alpha
         next_alpha_u = self.alpha_u[self.iteration]; 
         next_alpha_l = self.alpha_l[self.iteration];
      else: 
      #{
         #check if we satisfy the wolfe conditions
         wolfe_decrease = self.phi[self.iteration]<=(self.phi[0] + self.mu*self.phi_prime[0]*self.alpha_t[self.iteration])
         wolfe_curvature = abs(self.phi_prime[self.iteration])<=self.eta*abs(self.phi_prime[0])
         if (wolfe_decrease and wolfe_curvature):
            #declare victory
            next_alpha = self.alpha_t[self.iteration];
            return True, next_alpha

         #failsafe. sometimes we get -0.0 when walking toward alpha_min. 
         #we should have a better fix, but for now we will just yield if our
         #interval size goes to zero (but we think we have a lower bound and should
         #be able to converge
         if not (self.alpha_u[self.iteration]== -1):
            #we have a non-infinite interval. see if it is nonzero in size
            if (abs(self.alpha_t[self.alpha_l[self.iteration]] - self.alpha_t[self.alpha_u[self.iteration]]) < 1.0E-8 ):
               if (self.iprint>0): 
                  print "DEBUG: our inverval has effectively become zero likeley due to noise near the lower bound. conceding"
               next_alpha = self.alpha_t[self.iteration]
               return True, next_alpha;
         
         #we aren't done yet (or the answer is an endpoint)
         if (self.mode == self.GSD_MODE_PSI): 
            #see if we should switch to PHI mode
            if ( (self.psi[self.iteration]<=0.0) and (self.phi_prime[self.iteration]>0.0)) :
               self.mode = self.GSD_MODE_PHI;
         
         #TODO rewrite this mode block to decrease redundant code after we know it works
         if (self.mode == self.GSD_MODE_PSI):
         #{
            U1 = self.psi[self.iteration] > self.psi[self.alpha_l[self.iteration]]
            U2 = (self.psi[self.iteration]<=self.psi[self.alpha_l[self.iteration]]) and (self.psi_prime[self.iteration]*(self.alpha_t[self.alpha_l[self.iteration]] - self.alpha_t[self.iteration])>0.0)
            #U3 = (psi(iter)<=psi(alpha_l(iter))) && (psi_prime(iter)*(alpha_t(alpha_l(iter)) - alpha_t(iter))<0.0)
            
            #update the interval
            if (U1):
               next_alpha_l = self.alpha_l[self.iteration]
               next_alpha_u = self.iteration 
            elif (U2):
               next_alpha_l = self.iteration
               next_alpha_u = self.alpha_u[self.iteration]
            else: #U3
               next_alpha_l = self.iteration;
               next_alpha_u = self.alpha_l[self.iteration];
            
            have_next_alpha = False
            enforce_min = False

            if (U2):
               #there is a possiblity that we have not updated alpha_u
               if (self.alpha_u[self.iteration]== -1):
                  #we haven't updated our upper bound yet
                  if ((self.alpha_max-1.0E-8)<self.alpha_t[self.iteration]):
                     #the solution to the line search is past alpha_max, which is where we are.
                     #this is as good as we get
                     next_alpha = self.alpha_t[self.iteration]
                     return True, next_alpha
                  
                  #we need to make progress toward alpha_max
                  next_alpha = min(self.alpha_t[self.iteration]+self.delta_max*(self.alpha_t[self.iteration]-self.alpha_t[self.alpha_l[self.iteration]]),self.alpha_max)
                  have_next_alpha = True
            elif (U1):
               #there is a possibility that we have not updated alpha_l
               if (self.alpha_l[self.iteration]==0):
                  #we haven't updated the lower bound yet
                  if (self.alpha_t[self.iteration]<(self.alpha_min+1.0E-8)):
                     #we are at alpha_min
                     #alpha* < alpha_min
                     #this is as good as we get
                     next_alpha = self.alpha_t[self.iteration];
                     return True, next_alpha
                  
                  #//we need to enforce progress toward alpha_min
                  enforce_min = True;
            
            #check for sufficient progress:
            #paper says if there isn't sufficient decrease over two trials, bisect
            #we are going to compare the next interval to the previous interval
            #see if we had an interval in the previous iteration
            ago = 1 #one iteration ago
            if ((self.iteration-ago) >=0):
               #we are sufficiently many iterations in
               if not ((self.alpha_u[self.iteration]-ago)== -1):
                  #we had a finite interval 'ago' iterations ago
                  if (abs(self.alpha_t[next_alpha_l] - self.alpha_t[next_alpha_u]) > self.delta_I*abs(self.alpha_t[self.alpha_l[self.iteration-ago]] - self.alpha_t[self.alpha_u[self.iteration-ago]])):
                     if (self.iprint>0): 
                        print "DEBUG: bisecting"
                     #we haven't been decreasing the interval properly. bisect
                     next_alpha = 0.5*(self.alpha_t[next_alpha_l] + self.alpha_t[next_alpha_u])
                     have_next_alpha = True
                  else:
                     if (self.iprint>0): 
                        print "DEBUG: sufficient Interval decrease satisfied. old = "+str(abs(self.alpha_t[self.alpha_l[self.iteration-ago]] - self.alpha_t[self.alpha_u[self.iteration-ago]]))+" new = "+str(abs(self.alpha_t[next_alpha_l] - self.alpha_t[next_alpha_u]))

            if not (have_next_alpha):
               #we need to interpolate to get next trial alpha
               #we are in psi mode so pass psi values and gradients
               a_l = self.alpha_t[self.alpha_l[self.iteration]]
               a_t = self.alpha_t[self.iteration]
               a_u = 77.7 if (self.alpha_u[self.iteration]==-1) else self.alpha_t[self.alpha_u[self.iteration]] #default doesn't matter because we won't use it
               f_l = self.psi[self.alpha_l[self.iteration]]
               f_t = self.psi[self.iteration]
               f_u = 77.7 if (self.alpha_u[self.iteration]==-1) else self.psi[self.alpha_u[self.iteration]] #default doesn't matter because we won't use it
               g_l = self.psi_prime[self.alpha_l[self.iteration]]
               g_t = self.psi_prime[self.iteration]
               g_u = 77.7 if (self.alpha_u[self.iteration]==-1) else self.psi_prime[self.alpha_u[self.iteration]] #default doesn't matter because we won't use it
               next_alpha = self.interpolate(a_l,a_t,a_u,f_l,f_t,f_u,g_l,g_t,g_u)
               have_next_alpha = True
            
            if (enforce_min):
               max_allowed = max(self.delta_min*self.alpha_t[self.iteration],self.alpha_min)
               if (next_alpha < self.alpha_min): #can interp do this? maybe for sufficiently nonzero alpha_min?
                  next_alpha = self.alpha_min
               elif (next_alpha > max_allowed):
                  next_alpha = max_allowed
         #} #end PSI mode
         elif (self.mode == self.GSD_MODE_PHI):
         #{
            #if we are in phi mode then we don't have to worry about alpha_min and alpha_max, so this should be less involved
            
            #a (analog of U1)
            if (self.phi[self.iteration] > self.phi[self.alpha_l[self.iteration]]):
               next_alpha_l = self.alpha_l[self.iteration] 
               next_alpha_u = self.iteration 
            #b (analog of U2)
            elif ((self.phi[self.iteration]<=self.phi[self.alpha_l[self.iteration]]) and (self.phi_prime[self.iteration]*(self.alpha_t[self.alpha_l[self.iteration]] - self.alpha_t[self.iteration])>0.0)):
               next_alpha_l = self.iteration;
               next_alpha_u = self.alpha_u[self.iteration]
            #c (analog of U3)
            else:
               next_alpha_l = self.iteration
               next_alpha_u = self.alpha_l[self.iteration]
            
            #if we didn't have bounds on I between alpha_min and alpha_max before this iteration, then we
            #will have them for the next. the same conditions that trigger PSI -> PHI trigger c, so no need
            #to worry about alpha_min, alpha_max
            
            #i've convinced myself that if we do not (this iter) have an alpha_u < alpha_max, then we 
            #will not actually use it to interpolate. we will fall into either case 1 or case 2 for interp
            
            #interpolate based on phi to get the next trial alpha UNLESS we fail sufficient interval length reduction
            have_next_alpha = False
            #check for sufficient progress:
            #paper says if there isn't sufficient decrease over two trials, bisect
            #we are going to compare the next interval to the previous interval
            #see if we had an interval in the previous iteration
            ago = 1; #one iteration ago
            if ((self.iteration-ago) >=0):
               #we are sufficiently many iterations in
               if not (self.alpha_u[self.iteration-ago]== -1):
                  #we had a finite interval 'ago' iterations ago
                  if (abs(self.alpha_t[next_alpha_l] - self.alpha_t[next_alpha_u]) > self.delta_I*abs(self.alpha_t[self.alpha_l[self.iteration-ago]] - self.alpha_t[self.alpha_u[self.iteration-ago]])):
                     if (self.iprint>0): 
                        print "DEBUG: bisecting"
                     #we haven't been decreasing the interval properly. bisect
                     next_alpha = 0.5*(self.alpha_t[next_alpha_l] + self.alpha_t[next_alpha_u])
                     have_next_alpha = True
                  else:
                     if (self.iprint>0): 
                        print "DEBUG: sufficient Interval decrease satisfied. old = "+str(abs(self.alpha_t[self.alpha_l[self.iteration-ago]] - self.alpha_t[self.alpha_u[self.iteration-ago]]))+" new = "+str(abs(self.alpha_t[next_alpha_l] - self.alpha_t[next_alpha_u]))
            
            if not (have_next_alpha):
               #we need to interpolate to get next trial alpha
               #we are in phi mode so pass phi values and gradients
               a_l = self.alpha_t[self.alpha_l[self.iteration]]
               a_t = self.alpha_t[self.iteration]
               a_u = 77.7 if (self.alpha_u[self.iteration]==-1) else self.alpha_t[self.alpha_u[self.iteration]] #default doesn't matter because we won't use it
               f_l = self.phi[self.alpha_l[self.iteration]]
               f_t = self.phi[self.iteration]
               f_u = 77.7 if (self.alpha_u[self.iteration]==-1) else self.phi[self.alpha_u[self.iteration]] #default doesn't matter because we won't use it
               g_l = self.phi_prime[self.alpha_l[self.iteration]]
               g_t = self.phi_prime[self.iteration]
               g_u = 77.7 if (self.alpha_u[self.iteration]==-1) else self.phi_prime[self.alpha_u[self.iteration]] #default doesn't matter because we won't use it
               next_alpha = self.interpolate(a_l,a_t,a_u,f_l,f_t,f_u,g_l,g_t,g_u)
               have_next_alpha = True
         #} #end PHI mode
         else:
            print "something is wrong with ls internal variables"
            sys.exit(1)
      #} #end else {iteration != 0}
      
      #DEBUG - remove eventually
      if ( (next_alpha < (self.alpha_min-1.0E-8)) or (next_alpha > (self.alpha_max+1.0E-8)) ):
         print "ERROR: alpha_min = "+str(self.alpha_min)+"  alpha_max = "+str(self.alpha_max)+"  next_alpha = "+str(next_alpha)+"  WTF!"
         print "something exploded. split the difference between lower and upper"
         next_alpha = min(self.alpha_t[next_alpha_u],self.alpha_t[next_alpha_l]) + 0.5*(max(self.alpha_t[next_alpha_u],self.alpha_t[next_alpha_l]) - min(self.alpha_t[next_alpha_u],self.alpha_t[next_alpha_l]))
      if not (self.alpha_u[self.iteration]==-1):
         #check that we obey interval bounds
         if ((next_alpha < (min(self.alpha_t[next_alpha_u],self.alpha_t[next_alpha_l])-1.0E-8)) or (next_alpha > (max(self.alpha_t[next_alpha_u],self.alpha_t[next_alpha_l])+1.0E-8))):
            print "ERROR: next_alpha_l = "+str(self.alpha_t[next_alpha_l])+" next_alpha = "+str(next_alpha)+" next_alpha_u = "+str(self.alpha_t[next_alpha_u])+"   WTF!"
            print "Split the difference between lower and upper"
            next_alpha = min(self.alpha_t[next_alpha_u],self.alpha_t[next_alpha_l]) + 0.5*(max(self.alpha_t[next_alpha_u],self.alpha_t[next_alpha_l]) - min(self.alpha_t[next_alpha_u],self.alpha_t[next_alpha_l]))

      self.alpha_t = np.append(self.alpha_t,next_alpha) 
      self.alpha_u = np.append(self.alpha_u,next_alpha_u) 
      self.alpha_l = np.append(self.alpha_l,next_alpha_l)
      self.iteration = self.iteration+1
      return False, next_alpha
   #}
      
   #   uses an interpolation strategy to generate the next trial alpha given
   #   alphas (a), values (f), and  gradients (g) at the current iteration's
   #   lower (l) trial (t) and upper (u) positions
   #
   #   finite values are not always available for upper quantities, but
   #   these will not be used by the algorithm until they are finite, so
   #   we just pass arbitrary values for upper quantities in this case
   #   
   #   returns float_next_alpha
   def interpolate(self,a_l,a_t,a_u,f_l,f_t,f_u,g_l,g_t,g_u):
   #{
      a_next = -1.0
      #Case 1
      if (f_t > f_l):
         a_c = self.cubic_interp(a_l,a_t,f_l,f_t,g_l,g_t) #f_l,f_t,g_l,g_t
         a_q = self.quadratic_interp_ffg(a_l,a_t,f_l,f_t,g_l) #f_l,f_t,g_l
         if (abs(a_c - a_l) < abs(a_q-a_l)):
            a_next = a_c
         else:
            a_next = 0.5*(a_q+a_c)
      #Case 2
      elif ((f_t<=f_l) and (g_t*g_l < 0.0)):
         a_c = self.cubic_interp(a_l,a_t,f_l,f_t,g_l,g_t) #f_l,f_t,g_l,g_t
         a_s = self.quadratic_interp_fgg(a_l,a_t,f_l,g_l,g_t) #f_l,g_l,g_t 
         if (abs(a_c - a_t)>=abs(a_s-a_t)):
            a_next = a_c
         else:
            a_next = a_s
      #Case 3
      elif ((f_t<=f_l) and (g_t*g_l >= 0.0) and (abs(g_t)<=abs(g_l))):
         #debug - do we have a finite upper bound on the interval this iteration?
         if (self.alpha_u[self.iteration]==-1): 
            print "I think this should be impossible"
            sys.exit(1)

         a_c = self.cubic_interp(a_l,a_t,f_l,f_t,g_l,g_t) #f_l,f_t,g_l,g_t
         a_s = self.quadratic_interp_fgg(a_l,a_t,f_l,g_l,g_t) #f_l,g_l,g_t 
         #the next interval will be a_t,a_u
         #see if a_c is in the right direction
         #the correct direction is toward a_u from a_t
         if ( (a_c-a_t)*(a_u-a_t) > 0.0 ): 
            #cubic is in the right direction, but is it better? 
            #thought is shorter is better because a_t has lower value
            if (abs(a_c - a_t) < abs(a_s - a_t)):
               a_next = a_c
            else:
               a_next = a_s
         else:
            #cubic is not in the right direction, pick a_s
            a_next = a_s
         
         #make sure we respect the interval bounds
         #and don't go too far toward a_u, which is higher value
         delta_3 = 0.66; #their value #TODO make member parameter?
         if (a_t > a_l):
            a_next = min(a_t + delta_3*(a_u-a_t),a_next)
         else:
            a_next = max(a_t + delta_3*(a_u-a_t),a_next)
      #Case 4
      else:
         #debug - do we have a finite upper bound on the interval this iteration?
         if (self.alpha_u[self.iteration]==-1):
            print "I think this should be impossible"
            sys.exit(1)
         
         #see what we can do
         #I think g_t and g_u must have opposite sign here, 
         #so the cubic should give us a point between the two
         a_next = self.cubic_interp(a_t,a_u,f_t,f_u,g_t,g_u) #f_u,f_t,g_u,g_t
      
      return a_next
   #}

   #   These functions assume variables with _1 all apply to the same point and the same for _2, but
   #   There is no assumed relationship between values for _1 and _2
   #   Called by interpolate
   #   return floats_for_alpha
   def cubic_interp(self,a_1,a_2,f_1,f_2,g_1,g_2): #alpha_c
   #{

      a_next = -1.0
      
      #d = a_2 - a_1 = d(a_2)
      d = a_2 - a_1
      d2 = d*d
      d3 = d2*d
      #C(d) = c0 + c1*d + c2*d2 + c3*d3 = f_2
      #C'(d) = c1 + 2*c2*d + 3*c3*d2 = g_2
      #C(0) = c0 = f_1
      #C'(0) = c1 = g_1
      #cn coefficients dn = d^n
      c0 = f_1
      c1 = g_1
      c3 = (2.0*f_2 - d*g_2 - c1*d -2.0*c0)/(-1.0*d3)
      c2 = (g_2 - 3.0*d2*c3-c1)/(2.0*d)
      
      discriminant = 4.0*c2*c2 - 12.0*c1*c3
      
      if (discriminant >= 0.0):
         d_plus = (-2.0*c2 + math.sqrt(discriminant))/(6.0*c3)
         d_minus = (-2.0*c2 - math.sqrt(discriminant))/(6.0*c3)
         
         next_a_plus = d_plus + a_1
         next_a_minus = d_minus + a_1
         
         #we want the root corresponding to lower interpolated phi
         f_plus = c0 + c1*d_plus + c2*d_plus*d_plus + c3*d_plus*d_plus*d_plus
         f_minus = c0 + c1*d_minus + c2*d_minus*d_minus + c3*d_minus*d_minus*d_minus
         if (f_plus < f_minus):
            a_next = next_a_plus
         else:
            a_next = next_a_minus
      else: 
         if (self.iprint>0): 
            print "DEBUG: discriminant negative in cubic interp"
         a_next = self.quadratic_interp_fgg(a_1,a_2,f_1,g_1,g_2)

      return a_next
   #}

   def quadratic_interp_ffg(self,a_1,a_2,f_1,f_2,g_1): #alpha_q
   #{
      a_next = -1.0
      
      #d = a_2-a_1 = d(a_2)
      #Q(d) = c0 + c1*d + c2*d2
      #Q'(d) = c1 + 2*c2*d
      d = a_2 - a_1
      c0 = f_1
      c1 = g_1
      c2 = (f_2 - f_1 - g_1*d)/(d*d)
      
      a_next = a_1 + (-0.5*c1/c2)
      return a_next
   #}

   def quadratic_interp_fgg(self,a_1,a_2,f_1,g_1,g_2): #alpha_s
   #{
      a_next = -1.0
      
      #d = a_2-a_1 = d(a_2)
      #Q(d) = c0 + c1*d + c2*d2
      #Q'(d) = c1 + 2*c2*d
      d = a_2 - a_1
      c0 = f_1
      c1 = g_1
      c2 = (g_2 - g_1)/(2.0*d)

      a_next = a_1 + (-0.5*c1/c2)
      return a_next
   #}
#}
