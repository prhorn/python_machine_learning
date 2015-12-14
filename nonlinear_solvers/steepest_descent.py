import numpy as np
import sys
import math
from line_search import line_search
   
#simple steepest descent with line search

class steepest_descent:
#{

   #  comment - description of our update
   #  error - our current error
   #  the_problem - an object that knows how to perform problem-specific operations
      
   #  iprint            /**<Integer for controlling printing.*/
   #  value             /**<The value of the function.*/
   #  ls                /**<The line_search class object*/
   #  ls_param_pref     /**<Integer determining line search parameters*/
   #  iteration         /**<Current iteration. mostly to test if non-zero so that we don't do extra work*/
   #  direction         /**<The current direction that we are searching along*/

   def __init__(self,the_problem_,ls_param_pref_ = 2, iprint_=0):
   #{
      self.the_problem = the_problem_
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
      
      self.comment = "Line Search Step"
      
      if (self.iteration==0):
         #decide the first search direction
         self.direction = -1.0*grad
         self.comment = "New Steepest Descent Direction"
         #we are at the line search origin
         self.the_problem.ls_origin_to_current_pos()


      #see if we are done along the search direction
      #defining the univariate function phi(alpha)
      #double phi = value;

      phi_prime = np.dot(self.direction,grad)
      done_this_dir, next_alpha = self.ls.next_step(self.value,phi_prime)
      #if (done_this_dir):
      if (True):
         #we have either satisfied the wolfe conditions
         #at the current position or have deemed the
         #search to be fruitless
         
         #either way next_alpha is the alpha corresponding to
         #out current position
         
         #get our new search direction
         self.direction = -1.0*grad
         self.comment = "New Steepest Descent Direction"

         #reset the line search
         self.ls.new_line()
         #set the current position as the origin of
         #the next line search
         self.the_problem.ls_origin_to_current_pos()
      
         #we need to make a move so that we have an update this iteration
         #(we already know this point is not the global answer)
         
         #next_step for iteration 0 of the line search will always return false
         #but calling it here saves the gradient and value at the origin
         phi_prime = np.dot(self.direction,grad) #we have the same gradient but a new dir
         done_this_dir, next_alpha = self.ls.next_step(self.value,phi_prime)
      
      #cout << "next_alpha = " << setprecision(2) << scientific << next_alpha << endl; 
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
