import numpy as np
import sys
import math
from line_search import line_search
   
#simple steepest descent with line search

class l_bfgs:
#{

   #  comment - description of our update
   #  error - our current error
   #  the_problem - an object that knows how to perform problem-specific operations
      
   #  iprint            /**<Integer for controlling printing.*/
   #  value             /**<The value of the function.*/
   #  ls                /**<The line_search class object*/
   #  ls_param_pref     /**<Integer determining line search parameters*/
   #  iteration         /**< int. Current iteration. mostly to test if non-zero so that we don't do extra work*/
      
   #  max_vecs          /** int. the maximum number of of each s and y that we will save */
   #  n_vecs            /** int. the number of vectors (of each s and y) that we currently have saved*/
   #  latest_vec        /** int. the position in the fields of the latest vector added.
   #                         the vectors will be written at 0,1,2,..,max_vecs-1,0,1,...*/
   #  s                 /** list of vectors. s_k = x_{k+1} - x_k = alpha_k*dir_k */
   #  y                 /** list of vectors. y_k = grad_{k+1} - grad_k */
      
   
   #  direction         /**<The current direction that we are searching along*/
   #  prev_grad         /**<The gradient at the origin of the current line search*/

   def __init__(self,the_problem_,max_vecs_=20,ls_param_pref_ = 0, iprint_=0):
   #{
      self.the_problem = the_problem_
      self.max_vecs = max_vecs_
      self.iprint = iprint_
      self.ls_param_pref = ls_param_pref_
      self.ls = line_search(self.ls_param_pref,self.iprint)

      self.reset()
   #}
   
   def reset(self):
   #{
      self.ls.new_line()
      self.error = 777.77
      self.value = 777.77
      self.comment = " "
      self.iteration = 0
      self.direction = np.array([],dtype=np.float)
      self.prev_grad = np.array([],dtype=np.float)
      self.s = []
      self.y = []
      self.n_vecs = 0
      self.latest_vec = -1
   #}
   
   #returns True if we have converged
   def next_step(self):
   #{
      #get objective function value and gradient at current position
      self.value = self.the_problem.value() #returns float
      grad = self.the_problem.gradient() #returns np.array

      #empty gradient should trigger instant convergence
      if (grad.size == 0):
         self.error = 0.0;
         self.the_problem.ls_origin_to_current_pos(); #to old orbs
         return True
      
      #see if we have converged
      self.error = math.sqrt(np.dot(grad,grad)/float(grad.size))
      if (self.error < self.the_problem.tolerance):
         #set the origin of the line search to the current
         #position (make all variables in the_problem
         #consistent) and declare victory
         self.the_problem.ls_origin_to_current_pos()
         return True
      
      if (self.iteration==0):
         #decide the first search direction (sd)
         self.direction = -1.0*grad
         #we are at the line search origin
         self.prev_grad = grad
         self.the_problem.ls_origin_to_current_pos()

      self.comment = "Line Search Step"

      #see if we are done along the search direction
      #defining the univariate function phi(alpha)
      #double phi = value;

      phi_prime = np.dot(self.direction,grad)
      done_this_dir, next_alpha = self.ls.next_step(self.value,phi_prime)
      if (done_this_dir):
      #{
         #we have either satisfied the wolfe conditions
         #at the current position or have deemed the
         #search to be fruitless
         #
         #either way next_alpha is the alpha corresponding to
         #our current position

         #store the new data as temps until we know if we want
         #to keep it
         cur_y = grad-self.prev_grad
         cur_s = next_alpha*self.direction
         ys = np.dot(cur_y,cur_s)
         if (ys>0):
            #this pair is acceptable.
            #collect the latest gradient information
            if (self.n_vecs < self.max_vecs):
               self.n_vecs = self.n_vecs+1
            
            #determine where our new vectors should be placed
            if (self.latest_vec == self.max_vecs-1):
               self.latest_vec = 0 #loop back to the beginning
            else:
               self.latest_vec = self.latest_vec+1
            
            #store our current vectors
            if (len(self.s) <self.max_vecs):
               #we don't yet have a full list. append
               self.s.append(cur_s)
               self.y.append(cur_y)
            else:
               #our lists are max size. replace old values
               self.s[self.latest_vec] = cur_s
               self.y[self.latest_vec] = cur_y
         else:
            print "H information not updated because curvature violated"

         #get our new search direction
         if (self.n_vecs>0):
            #Nocedal p225
            #we have data with which to inform the next step
            gamma = np.dot(self.y[self.latest_vec],self.s[self.latest_vec])/np.dot(self.y[self.latest_vec],self.y[self.latest_vec])
            self.direction = grad 
            alpha = np.zeros(self.n_vecs,dtype=np.float)
            rho = np.zeros(self.n_vecs,dtype=np.float)
            #we apply most recent first
            cur_index = self.latest_vec
            for i in range(self.n_vecs):
               rho[cur_index] = 1.0/np.dot(self.y[cur_index],self.s[cur_index])
               alpha[cur_index] = np.dot(self.s[cur_index],self.direction)*rho[cur_index]
               self.direction = self.direction - alpha[cur_index]*self.y[cur_index]
               cur_index = cur_index - 1
               if (cur_index==-1): 
                  cur_index = self.max_vecs-1 #loop around
            self.direction = self.direction*gamma #our iteration specific Hzero
            #apply most recent last
            if ((self.n_vecs == self.max_vecs) and (self.latest_vec!=(self.max_vecs-1))): 
               cur_index = self.latest_vec+1 #loop around
            else:                    
               cur_index = 0
            for i in range(self.n_vecs):
               beta = rho[cur_index]*np.dot(self.y[cur_index],self.direction)
               self.direction = self.direction + (alpha[cur_index]-beta)*self.s[cur_index]
               cur_index = cur_index + 1
               if (cur_index==self.max_vecs): 
                  cur_index = 0 #loop around
            #dir is now (invHess)*grad
            self.direction = -1.0*self.direction
            self.comment = "BFGS Step"
            #we have our descent direction
         else:
            #we are stuck doing steepest descent 
            self.direction = -1.0*grad
            self.comment = "New Steepest Descent Direction"
         
         #we have our new search direction 
         #reset the line search
         self.ls.new_line()
         #set the current position as the origin of
         #the next line search
         self.the_problem.ls_origin_to_current_pos()
         #set the current gradient as the gradient at
         #the origin of the next line search
         self.prev_grad = grad
      
         #we need to make a move so that we have an update this iteration
         #(we already know this point is not the global answer)
         
         #next_step for iteration 0 of the line search will always return false
         #but calling it here saves the gradient and value at the origin
         phi_prime = np.dot(self.direction,grad) #we have the same gradient but a new dir
         done_this_dir, next_alpha = self.ls.next_step(self.value,phi_prime)
      #}      
      
      #update our position
      #go to the origin of the line search
      self.the_problem.move_to_ls_origin()
      #apply the suggested alpha
      disp = self.direction*next_alpha
      self.the_problem.update(disp)
      self.iteration = self.iteration+1
      return False
   #}
#}
