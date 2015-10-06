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

########
print 'antihypertensive drug example'
drug = (np.random.randn(100)*10.0) + 81.0
print 'mean of group given drug ',sample_mean(drug)
print 'standard deviation of group given drug ',sample_stdev(drug)
drug_interval = confidence_interval_mean(drug,95)
print '95% confidence interval for drug mean = ('+str(drug_interval[0])+','+str(drug_interval[1])+')'
placebo = (np.random.randn(100)*10.0) + 85.0
print 'mean of group given placebo ',sample_mean(placebo)
print 'standard deviation of group given placebo ',sample_stdev(placebo)
placebo_interval = confidence_interval_mean(placebo,95)
print '95% confidence interval for placebo mean = ('+str(placebo_interval[0])+','+str(placebo_interval[1])+')'
t,p_value,mean_diff_interval = t_test(drug,placebo,True,95)
print 't and p-values for test: '+str(t)+'  '+str(p_value)
print '95% confidence interval on difference between drug and placebo means = ('+str(mean_diff_interval[0])+','+str(mean_diff_interval[1])+')'


