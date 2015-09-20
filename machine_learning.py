import numpy as np
import sys

def mean_squared_error(Y_predicted,Y_actual):
#{
   #Y is assumed to be (observations) by (output quantities)
   #returns a vector (output quantities) long with mean squared
   #error for each output that we attempted to predict
   
   #doesn't actually matter which is predicted and which is acutal 
   #fxn is symmetric
   
   if not (Y_predicted.shape[0] == Y_actual.shape[0]):
      print 'number of rows of predicted and actual Y did not match'
      sys.exit(1)
   if not (Y_predicted.shape[1] == Y_actual.shape[1]):
      print 'number of cols of predicted and actual Y did not match'
      sys.exit(1)
   error = Y_predicted - Y_actual
   mse = np.zeros(Y_actual.shape[1])
   for i in range(Y_actual.shape[1]):
      mse[i] = np.dot(error[:,i],error[:,i])
   mse = (1.0/float(Y_actual.shape[0]))*mse
   return mse
#}

def classification_error_rate(Y_predicted,Y_actual):
#{
   if not (Y_predicted.shape[0] == Y_actual.shape[0]):
      print 'number of rows of predicted and actual Y did not match'
      sys.exit(1)
   if not (Y_predicted.shape[1] == Y_actual.shape[1]):
      print 'number of cols of predicted and actual Y did not match'
      sys.exit(1)
   diff = Y_predicted - Y_actual
   err = np.zeros(Y_actual.shape[1])
   for r in range(diff.shape[0]):
      for c in range(diff.shape[1]):
         if not (diff[r,c] ==0):
            err[c] = err[c] + 1.0
   err = err*(1.0/float(diff.shape[0]))
   return err
#}

def linear_regression(X,Y,include_intercept):
#{
   #solve B = (XtX)^-1 Xt Y
   #X and Y are numpy arrays, and include_intercept is a bool 

   N = X.shape[0] #rows of X, number of observations
   if not (Y.shape[0] == N):
      print "number of observations (rows) in X and Y did not match in linear_regression"
      sys.exit(1)
   
   if include_intercept:
      #augmented X
      aX = np.column_stack((X,np.ones(X.shape[0])))
   else:
      aX = X

   #compute and invert the metric
   metric = np.dot(aX.T,aX)
   U,s,V = np.linalg.svd(metric) #NB svals are decreasing
   tol = 1.0E-8
   n_indep = s.size
   for i in s:
      if (i < tol):
         n_indep -= 1

   if n_indep > 0:
      metric_inv = np.dot(U[:,:n_indep],np.dot(np.diag(s[:n_indep]**-1),np.transpose(U[:,:n_indep])))
   else:
      print 'poorly posed problem in linear_regression'
      sys.exit(1)
   
   B = np.dot(metric_inv,np.dot(aX.T,Y))

   return B
#} 

def linear_regression_train_test(X_train,Y_train,X_test,Y_test,param_tuple):
#{
   #fits a linear model for outputs (Y cols) with predictors in X (cols) 
   #plus an intercept based on training data and then computes
   #the mean squared error for the test set
   #param_tuple currently not used
   B = linear_regression(X_train,Y_train,True)
   predicted_Y = np.dot(np.column_stack((X_test,np.ones(X_test.shape[0]))),B) #add ones col to X to get intercept contriubtion
   mse = mean_squared_error(predicted_Y,Y_test) 
   return mse
#} 

def ridge_regression(X,Y,lambda_rr):
#{
   #lambda_rr = ridge regression shrinkage parameter
   if (lambda_rr<0.0):
      print 'shrinkage parameter cannot be negative in ridge regression'
      sys.exit(1)
   X_centered, mu = center_columns(X)
   Y_centered, intercepts = center_columns(Y)
   metric = np.dot(X_centered.T,X_centered) + np.diag(np.ones(X.shape[1])*lambda_rr)
   U,s,V = np.linalg.svd(metric) #NB svals are decreasing
   tol = 1.0E-8
   n_indep = s.size
   for i in s:
      if (i < tol):
         n_indep -= 1

   if n_indep > 0:
      metric_inv = np.dot(U[:,:n_indep],np.dot(np.diag(s[:n_indep]**-1),np.transpose(U[:,:n_indep])))
   else:
      print 'poorly posed problem in linear_regression'
      sys.exit(1)
   
   B = np.dot(metric_inv,np.dot(X_centered.T,Y_centered))
   B = np.vstack((B,intercepts))
   return B,mu 
#}

def ridge_regression_predict(X_predict,B,mu):
#{
   X_shift = np.array(X_predict)
   for i in range(X_shift.shape[1]):
      X_shift[:,i] -= mu[i]
   Y_predict = np.dot(np.column_stack((X_shift,np.ones(X_shift.shape[0]))),B)
   return Y_predict
#}

def ridge_regression_train_test(X_train,Y_train,X_test,Y_test,param_tuple):
#{
   #fits ridge regression for outputs (Y cols) with predictors in X (cols) 
   #plus an intercept based on training data and lambda in param_tuple and then computes
   #the mean squared error for the test set
   B,mu = ridge_regression(X_train,Y_train,param_tuple)
   predicted_Y = ridge_regression_predict(X_test,B,mu) 
   mse = mean_squared_error(predicted_Y,Y_test) 
   return mse
#} 

def n_cross_validation(training_inputs,training_outputs,model_error_function,param_tuple,n):
#{
   #perform CV(n)
   #returns a vector of errors for each column of Y that is an average over n fits
   #model error function takes model training and test set data and returns a vector of mse for that fit

   #check for some possible issues with input
   if not type(n) is int:
      print 'please specify an integer number of divisions of the data in n_cross_validation'
      sys.exit(1)
   if n<2:
      print 'number of cross validation groups must be greater than 1'
      sys.exit(1)
   if not (training_inputs.shape[0] == training_outputs.shape[0]):
      print 'number of observations in training input and outputs does not match'
      sys.exit(1)
      
   #nothing apparently wrong with input
   #determine batch sizes
   batch_sizes = np.zeros(n,dtype=np.int) 
   base_size = training_inputs.shape[0]//n #'floor division' operator
   batch_sizes.fill(base_size)
   still_to_distrib = training_inputs.shape[0] - base_size*n
   for i in range(still_to_distrib):
      batch_sizes[i] += 1
   
   #double check out batch sizes
   if not (np.sum(batch_sizes) == training_inputs.shape[0]):
      print 'batch sizes were invalid. FIX ME'
      sys.exit(1)
   
   #now randomly divide the observations (rows) between them
   row_permutation = np.random.permutation(training_inputs.shape[0])
   #the first batch will test on the first batch-size permuted rows
   #and fit based on the other rows

   #permute the rows of input and output
   permuted_inputs = training_inputs[row_permutation,:]
   permuted_outputs = training_outputs[row_permutation,:]
      
   #columns will corresponds to the errors for each of the Y outputs predicted
   trial_results = np.zeros((n,training_outputs.shape[1]))
   
   #notes on pointer math:
   #sum of zero dim array is 0, which is nice
   #n:m means return m-n elements starting with the nth (indexed at 0)
   #:m effective n=0
   #n: effective m=size; NB a[m] is out of bounds. a[m-1] is last elem
   for i in range(n):
      batch_before = np.sum(batch_sizes[:i]) #the offset for the ith fit's test rows in permuted matrices
      batch_after = np.sum(batch_sizes[i+1:]) #the number of rows after the ith fit's test rows in permuted matrices
      #get the ith set of data to fit 
      if (batch_before == 0) and (batch_after == 0):
         print 'we were fed garbage or pointer math went horribly wrong'
         sys.exit(1)
      elif batch_before == 0:
         #no rows before
         train_in_i = permuted_inputs[batch_before+batch_sizes[i]:,:]
         train_out_i = permuted_outputs[batch_before+batch_sizes[i]:,:]
      elif batch_after == 0:
         #no rows after
         train_in_i = permuted_inputs[:batch_before,:]
         train_out_i = permuted_outputs[:batch_before:,:]
      else:
         train_in_i = np.vstack((permuted_inputs[:batch_before,:],permuted_inputs[batch_before+batch_sizes[i]:,:]))
         train_out_i = np.vstack((permuted_outputs[:batch_before,:],permuted_outputs[batch_before+batch_sizes[i]:,:]))
      #get the ith set of data to test on
      test_in_i = permuted_inputs[batch_before:batch_before+batch_sizes[i],:]
      test_out_i = permuted_outputs[batch_before:batch_before+batch_sizes[i],:]
      #fit the model to this data
      mse_i = model_error_function(train_in_i,train_out_i,test_in_i,test_out_i,param_tuple)
      trial_results[i,:] = mse_i

   #we now have the errors for each of the n fits
   #average over fits
   avg_result = np.zeros(training_outputs.shape[1])
   for i in range(training_outputs.shape[1]):
      avg_result[i] = np.sum(trial_results[:,i])/float(n)
   return avg_result
#} 


def k_nearest_neighbors(X_train,Y_train,X_predict,k):
#{
   #returns our predictions for Y at the values X_predict based on the test data
   if not (X_train.shape[0] == Y_train.shape[0]):
      print 'number of observations in training data matrices did not agree in k_nearest_neighbors'
      sys.exit(1)
   if not (X_train.shape[1] == X_predict.shape[1]):
      print 'number of predictors (columns of X) inconsistent between predict and train'
      sys.exit(1)
   if (not (type(k) is int)) or (k<1):
      print 'k must be a positive integger in k_nearest_neighbors'
      sys.exit(1)
   if (k>X_train.shape[0]):
      print 'you asked for more neighbors than you provided training points. Try again'
      sys.exit(1)

   #determin the distance between each X point (observation) in train and in test
   sq_distances = np.zeros((X_train.shape[0],X_predict.shape[0])) #square distances will still give us the same ordering
   for i in range(X_train.shape[0]):
      for j in range(X_predict.shape[0]):
         disp_ij = X_train[i,:] - X_predict[j,:]
         sq_distances[i,j] = np.dot(disp_ij,disp_ij)
   
   Y_predict = np.zeros((X_predict.shape[0],Y_train.shape[1]))
   for j in range(X_predict.shape[0]):
      sort_indices = np.argsort(sq_distances[:,j]) #sorts increasing 
      sorted_output = Y_train[sort_indices[:k],:] #only grab the first k (smallest dist) post-sorting rows
      for c in range(Y_train.shape[1]):
         Y_predict[j,c] = np.sum(sorted_output[:,c])/float(k)
   
   return Y_predict
#}

def k_nearest_neighbors_train_test(X_train,Y_train,X_test,Y_test,param_tuple):
#{
   #param_tuple should be k, a positve integer
   #returns the mean squared error for the test set
   Y_predict = k_nearest_neighbors(X_train,Y_train,X_test,param_tuple)
   mse = mean_squared_error(Y_predict,Y_test) 
   return mse
#}

def k_nearest_neighbors_classification(X_train,Y_train,X_predict,k):
#{
   #returns our predictions for Y at the values X_predict based on the test data
   #Y_train is assumed a single column of integers corresponding to the different classification groups
   if not (Y_train.shape[1] == 1):
      print 'Too many columns in Y. Only implemented for classification into one set of groups.'
      sys.exit(1)
   if not (X_train.shape[0] == Y_train.shape[0]):
      print 'number of observations in training data matrices did not agree in k_nearest_neighbors'
      sys.exit(1)
   if not (X_train.shape[1] == X_predict.shape[1]):
      print 'number of predictors (columns of X) inconsistent between predict and train'
      sys.exit(1)
   if (not (type(k) is int)) or (k<1):
      print 'k must be a positive integger in k_nearest_neighbors'
      sys.exit(1)
   if (k>X_train.shape[0]):
      print 'you asked for more neighbors than you provided training points. Try again'
      sys.exit(1)

   #determin the distance between each X point (observation) in train and in test
   sq_distances = np.zeros((X_train.shape[0],X_predict.shape[0])) #square distances will still give us the same ordering
   for i in range(X_train.shape[0]):
      for j in range(X_predict.shape[0]):
         disp_ij = X_train[i,:] - X_predict[j,:]
         sq_distances[i,j] = np.dot(disp_ij,disp_ij)
   
   group_designations = np.unique(Y_train)
   Y_predict = np.ndarray((X_predict.shape[0],1),dtype=np.int)
   for j in range(X_predict.shape[0]):
      sort_indices = np.argsort(sq_distances[:,j]) #sorts increasing 
      sorted_output = Y_train[sort_indices[:k],:] #only grab the first k (smallest dist) post-sorting rows
      group_counts = np.ndarray(group_designations.size,dtype=np.int)
      group_counts.fill(0)
      for i in sorted_output:
         for g in range(len(group_designations)):
            if (i == group_designations[g]):
               group_counts[g] = group_counts[g]+1
      g_index = np.argmax(group_counts) #figure out which group has the most votes, ignoring tie-breaking for now
      Y_predict[j] = group_designations[g_index] 
   
   return Y_predict
#}

def k_nearest_neighbors_classification_train_test(X_train,Y_train,X_test,Y_test,param_tuple):
#{
   #param_tuple should be k, a positve integer
   #returns the mean squared error for the test set
   Y_predict = k_nearest_neighbors_classification(X_train,Y_train,X_test,param_tuple)
   err = classification_error_rate(Y_predict,Y_test) 
   return err
#}

def center_columns(X):
#{
   mu = np.mean(X, axis=0)
   X_centered = np.array(X)
   for i in range(X.shape[1]):
      X_centered[:,i] -=  mu[i]
   return X_centered, mu
#}

def standardize_columns(X):
#{
   X_centered,mu = center_columns(X)
   X_standardized = np.array(X_centered)
   stdev = np.std(X_centered,axis=0) 
   for i in range(X.shape[1]):
      X_standardized[:,i] = X_centered[:,i]/stdev[i] 
   return X_standardized, mu, stdev
#}

def lin_quad_discriminant_analysis_train(X,Y,is_LDA=True,do_regularization=False,alpha=1.0):
#{
   #INPUT:
   #  X:                   training predictors
   #  Y:                   (training outs) is assumed to have labels for k classes with integer values between 0 and k-1
   #  is_LDA:              if true, do LDA instead of QDA. only one sigma is prduced in this case and regularization is irrelevant
   #  do_regularization:   if true, make use of alpha if we are QDA as well in determination of covariance matrices (sigmas)
   #  alpha:               smaller alpha makes QDA more like LDA -- sigma_k = alpha*sigma_k + (1-alpha)*sigma

   #OUTPUT:
   #  mu:      the class means of the p columns of X in a k by p matrix
   #  pi:      vector of class priors (k of them)
   #  sigma:   list of covarance matrices, either 1 for LDA or k for QDA 
   
   #checks of input 
   if not (Y.shape[1] == 1):
      print 'Too many columns in Y. Only implemented for classification into one set of groups.'
      sys.exit(1)
   if not (X.shape[0] == Y.shape[0]):
      print 'number of observations in training data matrices did not agree'
      sys.exit(1)
   if do_regularization:
      if (alpha < 0.0) or (alpha > 1.0):
         print 'alpha must be between 0 and 1'
         sys.exit(1)
   
   #we need to do within-group computations. sort our training data by group
   sort_indices = np.argsort(Y[:,0])
   Y_sorted = Y[sort_indices,:]
   X_sorted = X[sort_indices,:]
   n_each_group = np.bincount(Y_sorted.reshape(Y_sorted.size)) 
   k = n_each_group.size

   #compute group priors
   pi = n_each_group / float(X_sorted.shape[0])

   #compute gaussian centers
   mu = np.zeros((n_each_group.size,X_sorted.shape[1]))
   for g in range(k):
      if n_each_group[g] > 0:
         obs_for_g = X_sorted[sum(n_each_group[:g]):sum(n_each_group[:g+1]),:]
         mu[g,:] = np.mean(obs_for_g, axis=0)
      else:
         print 'TOOD can we cleanly deal with the case of having no samples from a group?'
         sys.exit(1)
   
   #compute covariance matrices
   #TODO we could probably make this more efficient by combining mu and sigma calcualtions
   
   #compute the sigma of LDA if we need it
   if is_LDA or do_regularization: #if we are doing regularized, then we need the LDA result
      sig_lda = np.zeros((X_sorted.shape[1],X_sorted.shape[1]))
      for g in range(k):
         if n_each_group[g] > 0:
            #copy was necessary because we might re-center th columns again below if regularized
            obs_for_g_lda = np.array(X_sorted[sum(n_each_group[:g]):sum(n_each_group[:g+1]),:])
            for c in range(obs_for_g.shape[1]):
               obs_for_g_lda[:,c] -= mu[g,c]
            sig_lda += np.dot(obs_for_g_lda.T,obs_for_g_lda)
      sig_lda *= (1.0/float(X_sorted.shape[0] -k))
   
   #load up sigma
   sigma = []
   if is_LDA: 
      sigma.append(sig_lda)
   else:
      for g in range(k):
         if n_each_group[g] > 0:
            obs_for_g = X_sorted[sum(n_each_group[:g]):sum(n_each_group[:g+1]),:]
            for c in range(obs_for_g.shape[1]):
               obs_for_g[:,c] -= mu[g,c]
            sig_g = np.dot(obs_for_g.T,obs_for_g)
            sig_g *= (1.0/float(n_each_group[g]-1)) 
            if do_regularization:
               sig_g = (alpha*sig_g) + ((1.0 - alpha)*sig_lda)
            sigma.append(sig_g) 
         else:
            print 'TOOD can we cleanly deal with the case of having no samples from a group?'
            sys.exit(1)
   return mu,pi,sigma
#}

def lin_quad_discriminant_analysis_predict(X_predict,mu,pi,sigma):
#{
   #Designed to be used with the output of lin_quad_discriminant_analysis_train
   #
   #INPUT:
   #  X_predict:  predictors for observations that we want class predictions for (n x p)
   #  mu:         predictor means for each of the k classes (k x p)
   #  pi:         class priors (k long vector)
   #  sigma:      class covariance matrices (1 if we did LDA, k if we did QDA)
   #OUTPUT:
   #  Y_predict:  our class predictions for the passed observations (labels 0 to k-1) 
   
   #check for consistence in inputs
   if not (X_predict.shape[1] == mu.shape[1]):
      print 'columns of X  (number of predictors) does not agree with the columns of mu (predictor class means)'
      sys.exit(1)
   if not (mu.shape[0] == pi.size):
      print 'rows of mu (predictor means for k classes) not equal ot the number of elements in pi (class priors)'
      sys.exit(1)
   if not ((len(sigma) == 1) or (len(sigma)==pi.size)):
      print 'number of sigmas passed (covariance matrices) not consistent with LDA (1) or QDA (k)'
      sys.exit(1)
   
   #get this out of the way
   ln_pi = np.log(pi)
   
   #we only have to invert one sigma if we are LDA, so do it now
   do_LDA = False
   if len(sigma) == 1:
      do_LDA = True   
   
   if do_LDA:
      #compute sigma_inv = inv(sigma[0])
      U,s,V = np.linalg.svd(sigma[0]) #NB svals are decreasing and rows of V are right evecs
      tol = 1.0E-8
      n_indep = s.size
      for i in s:
         if (i < tol):
            n_indep -= 1
      if n_indep > 0:
         sigma_inv = np.dot(U[:,:n_indep],np.dot(np.diag(s[:n_indep]**-1),np.transpose(U[:,:n_indep])))
      else:
         print 'we were pass a garbage covarance matrix -- no non-zero singular values'
         sys.exit(1)
   
   #compute discriminants
   discriminants = np.zeros((X_predict.shape[0],mu.shape[0])) #each row contains the discrimiant values for an observation
   for g in range(mu.shape[0]):
      #get the inverse of the covariance matrix for this group/class
      if do_LDA:
         #LDA
         #delta_g(x) = x.t sigma^-1 mu_g - 0.5* mu_g.t sigma^-1 mu_g + ln_pi_g
         sigma_g_inv = sigma_inv
         sigma_mu_g = np.dot(sigma_g_inv,np.transpose(mu[g,:]))
         discriminants[:,g] = np.dot(X_predict,sigma_mu_g) + (ln_pi[g] - 0.5*np.dot(mu[g,:],sigma_mu_g)) 
      else:
         #QDA
         #delta_g(x) = x.t sigma_g^-1 mu_g - 0.5* mu_g.t sigma_g^-1 mu_g - 0.5* x.t sigma_g^-1 x  + ln_pi_g -0.5*ln_det_sig_g  
         #invert sigma for this group
         U,s,V = np.linalg.svd(sigma[g]) #NB svals are decreasing and rows of V are right evecs
         tol = 1.0E-8
         n_indep = s.size
         for i in s:
            if (i < tol):
               n_indep -= 1
         if n_indep > 0:
            sigma_g_inv = np.dot(U[:,:n_indep],np.dot(np.diag(s[:n_indep]**-1),np.transpose(U[:,:n_indep])))
         else:
            print 'we were pass a garbage covarance matrix -- no non-zero singular values'
            sys.exit(1)
         #compute some intermediates
         ln_det_sig_g = np.log(np.linalg.det(sigma[g]))
         sigma_mu_g = np.dot(sigma_g_inv,np.transpose(mu[g,:]))
         x_sigma_g = np.dot(X_predict,sigma_g_inv)
         #jam out all but the quadratic term
         discriminants[:,g] = np.dot(x_sigma_g,np.transpose(mu[g,:])) + (ln_pi[g] - 0.5*np.dot(mu[g,:],sigma_mu_g)  - 0.5*ln_det_sig_g)
         #now we need the quadratic terms
         x_sigma_g *= (-0.5)
         for i in range(discriminants.shape[0]):
            discriminants[i,g] += np.dot(x_sigma_g[i,:],X_predict[i,:])
   #classify based on largest discriminant
   Y_predict = np.ndarray(buffer=np.argmax(discriminants,axis=1),shape=(X_predict.shape[0],1),dtype=np.int)
   
   return Y_predict
#}

def lin_quad_discriminant_analysis_train_test(X_train,Y_train,X_test,Y_test,param_tuple):
#{
   #param_tuple should be (is_LDA,do_regularization,alpha)
   #returns the mean squared error for the test set
   mu,pi,sigma = lin_quad_discriminant_analysis_train(X_train,Y_train,param_tuple[0],param_tuple[1],param_tuple[2])
   Y_predict = lin_quad_discriminant_analysis_predict(X_test,mu,pi,sigma)
   err = classification_error_rate(Y_predict,Y_test) 
   return err
#}
