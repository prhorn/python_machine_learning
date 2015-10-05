import numpy as np
import math
import scipy.stats

def sample_mean(x):
#{
   mu = np.sum(x)/float(x.size)
   return mu
#}

def sample_variance(x):
#{
   mu = sample_mean(x)
   var = np.dot(x - mu,x - mu)/float(x.size - 1)
   return var
#}

def sample_stdev(x):
#{
   var = sample_variance(x)
   return math.sqrt(var)
#}

def sample_standard_error_of_mean(x):
#{
   return sample_stdev(x)/mat.sqrt(float(len(x)))
#}

def compute_percentiles(x,percentile_list):
#{
   #lienar interpolation method
   x_sorted = np.sort(x)
   x_len = x_sorted.size
   x_percent_rank =  100.0/float(x_len) * (np.arange(x_len) + 0.5 )
   percentiles = []
   for p in percentile_list:
      if (p <= x_percent_rank[0]):
         percentiles.append(x_sorted[0])
      elif (p >= x_percent_rank[-1]):
         percentiles.append(x_sorted[-1])
      else:
         for r in range(len(x_percent_rank)-1):
            if (x_percent_rank[r] <= p) and (p<=x_percent_rank[r+1]):
               percentiles.append( x_sorted[r] + (p - x_percent_rank[r])/(x_percent_rank[r+1] - x_percent_rank[r])*(x_sorted[r+1] - x_sorted[r]))
               break #done with r
   return percentiles
#}

def F_test(list_of_data_series):
#{
   #the F-test assumes that the samples are 
   #  independent
   #  random
   #  from populations following a normal distribution
   #  from populations with the same variance even if the mean is different
   K = len(list_of_data_series)
   if (K<2):
      print 'F-test needs at least two series'
      return
   
   n = []
   mu = []
   for l in list_of_data_series:
      n.append(len(l))
      mu.append(sample_mean(l))
   all_data = np.concatenate(list_of_data_series)
   N = len(all_data)
   MU = sample_mean(all_data)

   var_between = 0.0 
   for i in range(K):
      var_between = var_between + n[i]*((mu[i]-MU)**2)
   var_between = var_between/float(K-1)

   var_within = 0.0
   for i in range(K):
      for_var_i = list_of_data_series[i] - mu[i]
      var_within = var_within + np.dot(for_var_i,for_var_i) 
   var_within = var_within/float(N-K)
   
   F = var_between/var_within
   
   F_dof = (K-1,N-K) #num,denom
   
   #CDF value at a given position is the probability that we will have
   #a draw from the distribution that is less than or equal to the given value.
   #In this case, we want to know the probability of obtaining an
   #F value greater than (or equal to) our given value
   #so we want 1-CDF = survival function (SF)
   p_value = scipy.stats.f.sf(F,F_dof[0],F_dof[1])

   return F,p_value
#}

def t_test(a,b):
#{
   n_a = len(a)
   n_b = len(b)
   mu_a = sample_mean(a)
   mu_b = sample_mean(b)
   var_a = sample_variance(a)
   var_b = sample_variance(b)
   pooled_variance = (float(n_a - 1)*var_a + float(n_b-1)*var_b)/float(n_a + n_b - 2)
   
   t = (mu_a - mu_b)/math.sqrt(pooled_variance/float(n_a) + pooled_variance/float(n_b))  
   t = abs(t)
   t_dof = n_a + n_b - 2

   p_value = scipy.stats.t.cdf(-1.0*t,t_dof) + scipy.stats.t.sf(t,t_dof) # prob below -t plus prob above +t
   return t,p_value
#}

def Holm_Sidak(list_of_data_series,list_of_comparison_pairs,alpha):
#{
   n_series = len(list_of_data_series)
   k = len(list_of_comparison_pairs) 

   n_i = []
   mu_i = []
   var_i = []
   for i in list_of_data_series:
      n_i.append(len(i))
      mu_i.append(sample_mean(i))
      var_i.append(sample_variance(i))
   
   var_within = sum(var_i)/float(n_series)
   #degrees of freedom are based on the number of degrees of freedom in the within group variance
   N = sum(n_i)
   dof = N - n_series
   
   p_vals = []
   #for each comparison, compute an unadjusted p-value
   for c in list_of_comparison_pairs:
      t_i = abs( mu_i[c[0]] - mu_i[c[1]])/math.sqrt(var_within*(1.0/float(n_i[c[0]]) + 1.0/float(n_i[c[1]])))
      p_i = scipy.stats.t.cdf(-1.0*t_i,dof) + scipy.stats.t.sf(t_i,dof) # prob below -t plus prob above +t
      p_vals.append(p_i)
   #sort the unadjusted p-values
   sorted_comparison_indices = np.argsort(p_vals)
   print 'these are the p-vals'
   print p_vals
   print 'these are the indices that order p-vals'
   print sorted_comparison_indices
   #0 fail 1 pass with position corresponding to the position of the comparison in list_of_comparison_pairs
   pass_fail = np.zeros(k,dtype=np.int) #initialize to failure so that we can just break below
   for i in range(k):
      cur_alpha = 1.0 - math.pow(1.0 - alpha,1.0/float(k-i))
      if (p_vals[sorted_comparison_indices[i]] < cur_alpha):
         pass_fail[sorted_comparison_indices[i]] = 1
      else:
         break #all subsequent tests fail
   
   return pass_fail,p_vals 
#}

