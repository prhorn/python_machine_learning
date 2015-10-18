#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from arima_utility import *
from arima import ARIMA
from  statsmodels.tsa import arima_model as am

time_steps = 2000

p = 0
d = 1
q = 2

np.random.seed(11)
print 'generating ARIMA('+str(p)+','+str(d)+','+str(q)+') time series'
z,phi,theta,sigma_a_sq = gen_arima_model(p,d,q,time_steps)
times = range(time_steps)

print '======================================='
print 'printing parameters for genrated model:'
print '     this is sigma_a_sq '+str(sigma_a_sq)
if (p>0):
   print '     this is phi'
   print phi
if (q>0):
   print '     this is theta'
   print theta
print '======================================='

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(times,z,s=10,c='b',marker="s",label='ARIMA('+str(p)+','+str(d)+','+str(q)+')')
plt.legend(loc='best');
plt.show()

if (d==0):
   #test with statsmodels
   model = am.ARMA(z,(p,q))
   result = model.fit(trend='c',method='mle')
   print 'parameters from statsmodels fitting'
   print result.params

#mix up the guess
#trying -S minimization because numerics seem better
guess_sigma_a_sq = sigma_a_sq
if (p>0):
   guess_phi = phi #+ 0.01*np.random.random_sample(p)
   #guess_phi = result.params[1:1+p]
else:
   guess_phi = np.empty(0)
if (q>0):
   guess_theta = theta #+ 0.01*np.random.random_sample(q)
   #guess_theta = -1.0*result.params[1+p:]
else:
   guess_theta = np.empty(0)

print '======================================='
print 'printing guess parameters:'
print '     this is guess_sigma_a_sq '+str(guess_sigma_a_sq)
if (p>0):
   print '     this is guess_phi'
   print guess_phi
if (q>0):
   print '     this is guess_theta'
   print guess_theta
print '======================================='

#DEBUG
#gamma_0 = (1.0 + guess_theta[0]*guess_theta[0] - 2.0*guess_theta[0]*guess_phi[0])/(1.0 - guess_phi[0]*guess_phi[0])
#print 'DEBUG: gamma_0: ',gamma_0
#temp_1 = 1.0/(gamma_0-1.0)
#temp_2 = (1.0 - math.pow(guess_theta[0],2*time_steps))/(1.0 - math.pow(guess_theta[0],2))
#temp_3 = np.array(([1.0,-1.0],[-1.0,gamma_0]))
#temp_4 = np.array(([guess_phi[0]*guess_phi[0],-1.0*guess_theta[0]*guess_phi[0]],[-1.0*guess_theta[0]*guess_phi[0],guess_theta[0]*guess_theta[0]] ))
#temp_D = temp_1*temp_3 + temp_2*temp_4
#print 'DEBUG: D'
#print temp_D

model = ARIMA(p,d,q)
model.train(z,guess_phi,guess_theta,guess_sigma_a_sq)

print '======================================='
print 'printing parameters for fitted model:'
print '     this is sigma_a_sq '+str(model.sigma_a_sq)
if (p>0):
   print '     this is phi'
   print model.phi
if (q>0):
   print '     this is theta'
   print model.theta
print 'this was mu ',model.mu
print '======================================='

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(model.a_hat)),model.a_hat,s=10,c='b',marker="s",label='ARIMA('+str(p)+','+str(d)+','+str(q)+') residuals')
plt.legend(loc='best');
plt.show()

#print 'randome walk: ARIMA(0,1,0)'
#plot_example_arima(0,1,0,time_steps)
