import numpy as np

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
#} //end linear_regression

def linear_regression_train_test(X_train,Y_train,X_test,Y_test):
#{
   #fits a linear model for outputs (Y cols) with predictors in X (cols) 
   #plus an intercept based on training data and then computes
   #the mean squared error for the test set
   B = linear_regression(X_train,Y_train,True)
   predicted_Y = np.dot(np.column_stack((X_test,np.ones(X_test.shape[0]))),B) #add ones col to X to get intercept contriubtion
   mse = mean_squared_error(predicted_Y,Y_test) 
   return mse
#} //end linear_regression_error

def n_cross_validation(training_inputs,training_outputs,model_error_function,n):
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
   permuted_inputs = np.zeros(training_inputs.shape)
   permuted_outputs = np.zeros(training_outputs.shape)
   for i in range(training_inputs.shape[0]):
      permuted_inputs[i,:] = training_inputs[row_permutation[i],:]
      permuted_outputs[i,:] = training_outputs[row_permutation[i],:]
      
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
      mse_i = model_error_function(train_in_i,train_out_i,test_in_i,test_out_i)
      trial_results[i,:] = mse_i

   #we now have the errors for each of the n fits
   #average over fits
   avg_result = np.zeros(training_outputs.shape[1])
   for i in range(training_outputs.shape[1]):
      avg_result[i] = np.sum(trial_results[:,i])/float(n)
   return avg_result

#} //end n_cross_validation
