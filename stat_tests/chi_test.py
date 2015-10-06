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


placebo = np.concatenate((np.ones(18,dtype=np.int),np.zeros(7,dtype=np.int) ) )
aspirin = np.concatenate((np.ones(6,dtype=np.int),np.zeros(13,dtype=np.int) ) )

print 'asprin data'
chi_sq,p_value = chi_squared([placebo,aspirin],2)
print 'chi_sq ',chi_sq
print 'p-value ',p_value

z,p_value = z_test(placebo,aspirin,False)
print 'z without yates ',z
print 'p-value ',p_value

z,p_value = z_test(placebo,aspirin,True)
print 'z with yates ',z
print 'p-value ',p_value



