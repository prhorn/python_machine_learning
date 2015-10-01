#! /usr/bin/env python

from quadratic_test import quadratic_test
from steepest_descent import steepest_descent

max_iter = 1000
problem_size = 50

problem = quadratic_test(50)
problem.obtain_guess()
optimizer = steepest_descent(problem)

converged = False
cur_iter = 0
for i in range(max_iter):
   cur_iter = cur_iter + 1
   converged = optimizer.next_step() 
   if (converged):
      break
   print "  "+str(cur_iter)+"  "+str(problem.value())+"  "+str(optimizer.error)+"  "+optimizer.comment

if (converged):
   print "  "+str(cur_iter)+"  "+str(problem.value())+"  "+str(optimizer.error)+"  Optimization Converged"
else:
   print 'optimization failed to converge'
