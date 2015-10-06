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
   return sample_stdev(x)/math.sqrt(float(len(x)))
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

def t_test(a,b,compute_confidence_interval_diff_mean=False,confidence_level=95):
#{
   #if compute_confidence_interval_diff_mean is true,
   #we compute the confidence_level% interval for the
   #difference between the two sample means

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

   #p_value = scipy.stats.t.cdf(-1.0*t,t_dof) + scipy.stats.t.sf(t,t_dof) # prob below -t plus prob above +t
   p_value = 2.0*scipy.stats.t.sf(t,t_dof) # symmetric 
   
   if not compute_confidence_interval_diff_mean:
      return t,p_value
   else:
      #-t_alpha < ((delta_sample_means) - (delta_true_means))/standard_error_of_difference_of_means < t_alpha
      alpha = 1.0 - confidence_level/100.0
      #compute the most extreme t-value we would accept
      #to contain confidence_level% of the t distribution,
      #remembering that it is symmetric
      t_alpha = scipy.stats.t.isf(alpha/2.0,t_dof) #inverse survival function
      lower_bound = (mu_a - mu_b) - t_alpha*math.sqrt(pooled_variance/float(n_a) + pooled_variance/float(n_b))
      upper_bound = (mu_a - mu_b) + t_alpha*math.sqrt(pooled_variance/float(n_a) + pooled_variance/float(n_b))
      return t,p_value,(lower_bound,upper_bound)
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
      #p_i = scipy.stats.t.cdf(-1.0*t_i,dof) + scipy.stats.t.sf(t_i,dof) # prob below -t plus prob above +t
      p_i = 2.0*scipy.stats.t.sf(t_i,dof) # symmetric
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
      cur_alpha = Sidak_adjusted_trial_alpha(alpha,k-i)
      if (p_vals[sorted_comparison_indices[i]] < cur_alpha):
         pass_fail[sorted_comparison_indices[i]] = 1
      else:
         break #all subsequent tests fail
   
   return pass_fail,p_vals 
#}

def z_test(a,b,apply_yates=True,compute_confidence_interval_diff_prop=False,confidence_level=95):
#{
   #values in the two samples, a and b,
   #are 0 or 1 corresponding to two states
   #we test to see if there is a significance 
   #difference in the probability of state 1
   #(and thus of state 0)
   #yates refers to the yates correction for continuity
   
   #if compute_confidence_interval_diff_prop is true,
   #we compute the confidence_level% interval for the
   #difference between the two sample proportions

   n_a = len(a)
   n_b = len(b)
   #proportion of observations in state 1 for series a
   p_a = float(sum(a))/float(n_a)
   p_b = float(sum(b))/float(n_b)
   #we use pooled information for variance = p_overall(1-p_overall)
   p_overall = (p_a*float(n_a) + p_b*float(n_b))/float(n_a + n_b)
   std_err_diff_p = math.sqrt(p_overall*(1.0-p_overall)*(1.0/float(n_a) + 1.0/float(n_b)))
   if apply_yates:
      z = (abs(p_a - p_b) - 0.5*(1.0/float(n_a) + 1.0/float(n_b)))/std_err_diff_p
   else:
      z = abs(p_a - p_b)/std_err_diff_p

   #z is normal
   #p_value = scipy.stats.norm.cdf(-1.0*z) + scipy.stats.norm.sf(z) # prob below -z plus prob above +z
   p_value = 2.0*scipy.stats.norm.sf(z) # symmetric
   
   if not compute_confidence_interval_diff_prop:
      return z,p_value
   else:
      #-z_alpha < ((delta_sample_proportions) - (delta_true_proportions))/standard_error_of_difference_of_proportions < z_alpha
      alpha = 1.0 - confidence_level/100.0
      #compute the most extreme values of the standard normal we would accept
      #to contain confidence_level% of the normal distribution,
      #remembering that it is symmetric
      z_alpha = scipy.stats.norm.isf(alpha/2.0) #inverse survival function
      lower_bound = (p_a - p_b) - z_alpha*std_err_diff_p
      upper_bound = (p_a - p_b) + z_alpha*std_err_diff_p
      return z,p_value,(lower_bound,upper_bound)
      
#}

def chi_squared(list_of_data_series,n_categories,print_table=False):
#{
   #we assume that the series contain integers
   #0, 1, 2,... n_categories-1
   
   n_series = len(list_of_data_series)
   #rows are categories. cols are series
   #entires are the number of observations
   #of that category in that series
   cat_ser = np.zeros(shape=(n_categories,n_series),dtype = np.int)
   for c in range(n_series):
      for i in list_of_data_series[c]:
         cat_ser[i,c] = cat_ser[i,c] + 1

   if print_table:
      print 'rows correspond to categories: 0,1,2,..n_categories-1'
      print 'columns correspond to the series passed'
      print cat_ser

   #number of observations in each series
   obs_series = []
   for i in range(n_series):
      obs_series.append(np.sum(cat_ser[:,i]))
   total_obs = sum(obs_series)
   
   #the probability across all series of being in a given category
   global_proportions = []
   for i in range(n_categories):
      global_proportions.append(float(np.sum(cat_ser[i,:]))/float(total_obs)) 

   chi_sq = 0.0
   for r in range(n_categories):
      for c in range(n_series):
         #(observed - expected)^2 / expected
         expected = global_proportions[r]*obs_series[c]
         chi_sq = chi_sq + ((cat_ser[r,c] - expected)**2)/expected

   dof = (n_categories - 1)*(n_series - 1)
   
   p_value = scipy.stats.chi2.sf(chi_sq,dof)

   return chi_sq,p_value
#}

def Sidak_adjusted_trial_alpha(overall_target_alpha,n_comparisons):
#{
   #return the alpha that we should compare an unadjusted p-value
   #of a single comparison to in order to keep the error rate for 
   #the collection of comparisons below overall_target_alpha
   return (1.0 - math.pow(1.0 - overall_target_alpha,1.0/float(n_comparisons)))
#}

def Sidak_adjust_p_value(unadjusted_p_value,n_comparisons):
#{
   #return the p_value that we should compare to the overall
   #target alpha for the set of comparisons given the unadjusted
   #p_value of a single comparison

   return (1.0 - ((1.0 - unadjusted_p_value)**n_comparisons))
#}

def confidence_interval_mean(data,confidence_level):
#{
   #if confidence_level = 95
   #then 95% of such confidence intervals produced from
   #samples of the same population will contain the
   #true population mean
   ###
   #data is a vector of observations from a single sample
   
   dof = len(data)-1
   alpha = 1.0 - confidence_level/100.0
   #compute the most extreme t-value we would accept
   #to contain confidence_level% of the t distribution,
   #remembering that it is symmetric
   t_alpha = scipy.stats.t.isf(alpha/2.0,dof) #inverse survival function
   
   # -t_alpha <= (sample_mean - true_mean)/sample_standard_error_of_mean  < t_alpha
   s = sample_standard_error_of_mean(data)
   mu = sample_mean(data)

   lower_bound = mu - t_alpha*s
   upper_bound = mu + t_alpha*s
   return (lower_bound,upper_bound)
#}

def confidence_interval_proportion(data,confidence_level):
#{
   #if confidence_level = 95
   #then 95% of such confidence intervals produced from
   #samples of the same population will contain the
   #true population proportion of category 1
   ###
   #data is a vector of integers 0 or 1 
   #p_1 + p_0 = 1
   #we are computing confidence interval for p1
   
   alpha = 1.0 - confidence_level/100.0
   #compute the most extreme values of the standard normal we would accept
   #to contain confidence_level% of the normal distribution,
   #remembering that it is symmetric
   z_alpha = scipy.stats.norm.isf(alpha/2.0) #inverse survival function
   
   #-z_alpha < (observed_proportion_1 - true_proportion_1)/standard_error_of_proportion < z_alpha
   p1 = sample_mean(data)
   s_p1 = math.sqrt((p1*(1.0-p1))/float(len(data))) #sample stdev = sqrt(p(1-p)) #standard error of proportion divides by sqrt(n)
   
   lower_bound = p1 - z_alpha*s_p1
   upper_bound = p1 + z_alpha*s_p1

   return (lower_bound,upper_bound)
#}


