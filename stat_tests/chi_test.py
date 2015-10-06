#! /usr/bin/env python

from simple_stats import *
import numpy as np


controls = np.concatenate((np.ones(14,dtype=np.int),np.zeros(40,dtype=np.int) ) )
joggers = np.concatenate((np.ones(9,dtype=np.int),np.zeros(14,dtype=np.int) ) )
runners = np.concatenate((np.ones(46,dtype=np.int),np.zeros(42,dtype=np.int) ) )

print 'runner data'
chi_sq,p_value = chi_squared([controls,joggers,runners],2,True)
print 'chi_sq',chi_sq
print 'p-value',p_value


print 'asprin data'
placebo = np.concatenate((np.ones(18,dtype=np.int),np.zeros(7,dtype=np.int) ) )
placebo_i = confidence_interval_proportion(placebo,95)
print 'placebo proportion with condition ',sample_mean(placebo)
print 'placebo proportion 95% confidence interval = ('+str(placebo_i[0])+','+str(placebo_i[1])+')'
aspirin = np.concatenate((np.ones(6,dtype=np.int),np.zeros(13,dtype=np.int) ) )
aspirin_i = confidence_interval_proportion(aspirin,95)
print 'aspirin proportion with condition ',sample_mean(aspirin)
print 'aspirin proportion 95% confidence interval = ('+str(aspirin_i[0])+','+str(aspirin_i[1])+')'

chi_sq,p_value = chi_squared([placebo,aspirin],2)
print 'chi_sq ',chi_sq
print 'p-value ',p_value

z,p_value,interval = z_test(placebo,aspirin,False,True,95)
print 'z without yates ',z
print 'p-value ',p_value
print 'proportion difference (placebo - aspirin) 95% confidence interval = ('+str(interval[0])+','+str(interval[1])+')'

z,p_value = z_test(placebo,aspirin,True)
print 'z with yates ',z
print 'p-value ',p_value



