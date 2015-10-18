import math
import numpy as np
import sys
from copy import *
import matplotlib.pyplot as plt

def gen_arima_model(p,d,q,steps,add_mean=False):
#{
   #steps will be the length of the time series z returned
   #we first generate a differenced time series w that is steps-d long
   
   #phi(B)w_t = theta(B)a_t

   unit_circle_buffer = 0.10
   #determine a valid phi(B)
   phi = []
   if (p>0):
      #phi(B) = \Pi_i (1-coef_i * B)
      #must have abs(coef_i)<1.0 for all i
      #abs val of coefs between buffer and 1.0-buffer
      product_coefs = (1.0-2.0*unit_circle_buffer)*np.random.random_sample(p) + unit_circle_buffer
      #make sign random
      product_coefs = product_coefs * np.random.choice([-1.0,1.0],p)
      #the roots of the phi polynomial are actually 1/product_coefs
      phi_roots = [1.0/i for i in product_coefs] 
      #generate coefficients for these roots
      temp_phi = np.poly(phi_roots)
      #the above temp_phi are phi_{p}, phi_{p-1}, phi_{p-2},...,phi_1,constant
      #with the coefficient for B^p, phi_{p}, = 1.0
      #we want to rescale so that the constant is 1.0
      temp_phi = temp_phi/float(temp_phi[-1])
      #we also need the order phi_1, phi_2,...,phi_p
      phi = [temp_phi[i] for i in reversed(range(len(temp_phi)-1))]

   #determine a valid theta(B)
   theta = []
   if (q>0):
      #theta(B) = \Pi_i (1-coef_i * B)
      #must have abs(coef_i)<1.0 for all i
      #abs val of coefs between buffer and 1.0-buffer
      product_coefs = (1.0-2.0*unit_circle_buffer)*np.random.random_sample(q) + unit_circle_buffer
      #make sign random
      product_coefs = product_coefs * np.random.choice([-1.0,1.0],q)
      #the roots of the theta polynomial are actually 1/product_coefs
      theta_roots = [1.0/i for i in product_coefs] 
      #generate coefficients for these roots
      temp_theta = np.poly(theta_roots)
      #the above temp_theta are theta_{q}, theta_{q-1}, theta_{q-2},...,theta_1,constant
      #with the coefficient for B^q, theta_{q}, = 1.0
      #we want to rescale so that the constant is 1.0
      temp_theta = temp_theta/float(temp_theta[-1])
      #we also need the order theta_1, theta_2,...,theta_q
      theta = [temp_theta[i] for i in reversed(range(len(temp_theta)-1))]
   
   #determine the variance for shocks (sigma_sq_a)
   #uniform 1,2
   #sigma_sq_a = np.random.random_sample() + 1.0
   sigma_sq_a = 1.0 
   
   #generated the differenced time series based on model params
   w = gen_arima_data(phi,theta,sigma_sq_a,steps-d)
   #print 'mean of generated w ',np.sum(w)/float(len(w))   
   #add a mean
   if (add_mean == 'if_d_zero'):
      if (d == 0):
         w_mu = w + np.random.random_sample()
   elif (add_mean):
      w_mu = w + np.random.random_sample()
   else:
      w_mu = w

   #invert differencing
   z = w_mu 
   for i in range(d):
      z = integrate_time_series(z,np.random.random_sample())

   return z,phi,theta,sigma_sq_a

#}

def integrate_time_series(w,origin):
#{
   #w is the  differenced series
   z = [origin] + [0.0]*len(w)
   for i in range(len(w)):
      z[i+1] = z[i] + w[i]
   return np.array(z)
#}

def gen_arima_data(phi,theta,sigma_sq_a,steps):
#{
   #ARMA(p,q)
   #w_t - \sum_{i=1}^{p} w_{t-i}\phi_{i} = a_t - \sum_{i=1}^{q} a_{t-i} \theta_{i}
   
   p = len(phi)
   q = len(theta)
   
   #we are going to run for longer than
   #asked so that our final values for
   #w aren't too dependent on our initial
   #values (zeros here)
   run_length = 3*(steps+p+q)

   #intialize w
   w = [0.0]*(p+run_length) #we need p past w's
   #get all shocks
   a = np.random.normal(0.0,math.sqrt(sigma_sq_a),run_length+q) #we need q past a's
   for t in range(run_length):
      #AR part
      for pp in range(p):
         w[p+t] = w[p+t] + w[p+t-(1+pp)]*phi[pp] 
      #MA part
      for qq in range(q):
         w[p+t] = w[p+t] - a[q+t-(1+qq)]*theta[qq] 
      #current shock
      w[p+t] = w[p+t] + a[q+t]
   
   #just return the end of the series that we generated
   return np.array(w[-steps:])
#}

def plot_example_arima(p,d,q,verbose=False,steps=2000):
#{
   print 'generating ARIMA('+str(p)+','+str(d)+','+str(q)+') time series'
   z,phi,theta,sigma_sq_a = gen_arima_model(p,d,q,steps)
   times = range(steps)
   
   if verbose:
      if (p>0):
         print 'this is phi'
         print phi
      if (q>0):
         print 'this is theta'
         print theta

   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.scatter(times,z,s=10,c='b',marker="s",label='ARIMA('+str(p)+','+str(d)+','+str(q)+')')
   plt.legend(loc='best');
   plt.show()
   return
#}
