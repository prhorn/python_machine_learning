#! /usr/bin/env python
import sys
import os
sys.path.append(os.getcwd()+'/nonlinear_solvers/')
from quadratic_test import quadratic_test
from steepest_descent import steepest_descent
from l_bfgs import l_bfgs

max_iter = 4000
problem_size = 10

problem = quadratic_test(problem_size)
problem.obtain_guess()
#optimizer = steepest_descent(problem)
optimizer = l_bfgs(problem)

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
