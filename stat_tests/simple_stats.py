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

   return F,F_dof,p_value
#}
