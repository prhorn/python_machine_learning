#! /usr/bin/env python

from simple_stats import *
import numpy as np

n = 200
alpha = 0.05

a = np.random.randn(n)
b = np.random.randn(n) + 0.2
c = np.random.randn(n) + 0.5
d = np.random.randn(n) + 1.0
samples = [a,b,c,d]

F_value,p_for_F =  F_test(samples)
print 'results of performing F-test on samples' 
print 'F was ',F_value
print 'p-value was ',p_for_F
F_success = p_for_F < alpha
if F_success:
   print 'F-test passed alpha = ',alpha
else:
   print 'F-test failed alpha = ',alpha

if F_success:
   pairs = [[0,1],[0,2],[0,3]]
   print 'performing Holm-Sidak Test for the following pairs '
   print pairs
   pass_fail,HS_p = Holm_Sidak(samples,pairs,alpha)
   print 'pass_fail '
   print pass_fail
   print 'unadjusted p-values for comparisons'
   print HS_p
   
   print 'computing standard t-test for all considered pairs'
   standard_t_p = []
   for i in pairs:
      t_i,p_i = t_test(samples[i[0]],samples[i[1]])
      standard_t_p.append(p_i)
   print 'standard t-test for all considered pairs'
   print standard_t_p
