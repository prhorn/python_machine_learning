import math

import numpy as np
import unittest
import scipy.stats

class LinearRegression:
   
    def __init__(self, include_intercept=True, compute_stats=True, confidence_level=95):
        self.include_intercept=include_intercept
        self.compute_stats = compute_stats
        self.confidence_level = confidence_level
        
    def train(self, X_train, Y_train):
        # input:
        #  X_train -- Nxm feature matrix or vector if m==1
        #  Y_train -- Nxp target matrix or vector if p==1
        #
        # output:
        #  self.B  -- mxp coefficient matrix or vector if p==1
        #  self.intercepts --   p-long vector of intercepts if
        #                       (self.include_intercept)
          
        # sort our dimensions of inputs
        N = X_train.shape[0]
        if N != Y_train.shape[0]:
            raise ValueError("Input X and Y did not have the same "
                             "number of observations")
        
        try:
            m = X_train.shape[1]
        except:
            m = 1
        
        try:
            p = Y_train.shape[1]
        except:
            p = 1
        
        # If we are including an intecept, then we need to augment
        # the passed X_train with a column of 1's as X
        # X will be a matrix after this step either way
        if self.include_intercept:
            X = np.column_stack((X_train,np.ones(N)))
        else:
            X = np.ndarray(buffer=X_train, shape=(N,m), dtype=np.float)
        
        # make Y a matrix as well so that B is consistently a matrix
        Y = np.ndarray(buffer=Y_train, shape=(N,p), dtype=np.float)
        
        # determine coefficients and intercepts
        # b = (X.T X)^-1 X.T Y
        metric = np.dot(X.T,X)
        metric_inv = np.linalg.pinv(metric)
        least_sq = np.dot(metric_inv,np.dot(X.T,Y))
        
        if self.include_intercept:
            self.intercepts = np.array(least_sq[-1,:])
            if p==1:
                self.B = np.array(least_sq[:-1,0])
            else:
                self.B = least_sq[:-1,:]
        else:
            if p==1:
                self.B = np.array(least_sq[:,0])
            else:
                self.B = least_sq
        
        # compute statistics about the fit if asked
        if self.compute_stats:
            # we store stats as members if we make them here
            # statistics will be in vectors p-long 
            # (the number of different quantities predicted)
            
            # get our least squares predictions
            Y_hat = np.dot(X,least_sq)
            
            # determine the threshold t-value for our coefficient confidence intervals
            alpha = 1.0 - float(self.confidence_level)/100.0
            t_dof = N - least_sq.shape[0]
            thresh_t_value = scipy.stats.t.isf(alpha/2.0, t_dof) 

            self.r_squared = []
            self.f_stat_mean = []
            self.p_value_mean = []
            self.parameter_stats = []
            for i in range(p):
                residual = Y[:,i] - Y_hat[:,i]
                ss_residual = np.dot(residual,residual)
                mean = np.sum(Y[:,i])/float(N)
                centered = Y[:,i] - mean
                # equivalent to the ss_residual for the mean model (intercept only)
                ss_tot = np.dot(centered,centered)
                # the coefficient of determination, r_squared
                self.r_squared.append(1.0 - ss_residual/ss_tot)
                
                # variance of the residuals (denominator accounts for intercept)
                residual_var = ss_residual/float(t_dof)
                # variances of the linear coefficients and intercepts
                parameter_vars = metric_inv.diagonal()*residual_var  
                
                # compute f-statistic for the mean model hypothesis
                self.f_stat_mean.append((ss_tot - ss_residual)/
                                        float(least_sq.shape[0] - 1)/residual_var)
                # the corresponding p-value for the mean model hypothesis
                self.p_value_mean.append(
                    scipy.stats.f.sf(self.f_stat_mean[i], least_sq.shape[0] - 1, t_dof))
                
                # statistics on our coefficients (and intercepts) will be collected as follows:
                # param_value t-value p-value lower_bound upper_bound
                stats_i = np.empty((least_sq.shape[0],5), dtype=np.float)
                # loop over parameters
                for j in range(least_sq.shape[0]): 
                    # parameter value
                    stats_i[j,0] = least_sq[j,i]
                    # t-value
                    stats_i[j,1] = least_sq[j,i]/math.sqrt(parameter_vars[j])
                    # p-value
                    stats_i[j,2] = 2.0*scipy.stats.t.sf(stats_i[j,1],t_dof)
                    # confidence interval lower bound
                    stats_i[j,3] = least_sq[j,i] - thresh_t_value*math.sqrt(parameter_vars[j])
                    # confidence interval upper bound
                    stats_i[j,4] = least_sq[j,i] + thresh_t_value*math.sqrt(parameter_vars[j])
                self.parameter_stats.append(stats_i)

    def predict(self,X_predict):
        # see if we have trained
        if not hasattr(self, 'B'): 
            raise RuntimeError("Model must be trained before predicting")
        
        N = X_predict.shape[0]
    
        try:
            m = X_predict.shape[1]
        except:
            m = 1

        # see if these dimensions are commensurate with
        # those used to train our coefficents
        if m != self.B.shape[0]:
            raise ValueError("Input X for prediction did not have " 
                             "the expected number of columns")
        
        Y_predict = np.dot(np.ndarray(buffer=X_predict, shape=(N,m)),self.B)
        
        # add intercept contriubtions if applicable
        if self.include_intercept:
            try:
                p = self.B.shape[1]
            except:
                p = 1
            
            if p == 1:
                Y_predict = Y_predict + self.intercepts[0]
            else:
                for i in range(p):
                    Y_predict[:,i] = Y_predict[:,i] + self.intercepts[i]
        
        return Y_predict


class LinearRegressionTests(unittest.TestCase):

    def test_single_predictor_single_response(self):
        N = 10
        coefs = np.array([[5.0]])
        X = np.random.rand(N)*10.0 
        Y = np.dot(np.ndarray(buffer=X, shape=(N,1)),coefs)
        # no random component
        
        # include intercept
        linreg = LinearRegression(True)

        # test train function
        linreg.train(X,Y)
        for i in range(coefs.shape[0]):
            self.assertAlmostEqual(
                coefs[i,0],linreg.B[i],
                msg="Model coefficients did not agree.")
        for i in range(len(linreg.r_squared)):
            self.assertAlmostEqual(
                1.0,linreg.r_squared[i],
                msg="Linear regression fit statistics inaccurate.")
        
        # test predict function
        Y_predict = linreg.predict(X)
        for i in range(Y.shape[0]):
            self.assertAlmostEqual(
                Y[i],Y_predict[i],
                msg="Predicted values did not agree.")
    
    def test_single_predictor_single_response_intercept(self):
        N = 10
        coefs = np.array([[5.0]])
        intercepts = np.array([1.0])
        X = np.random.rand(N)*10.0 
        Y = np.dot(np.ndarray(buffer=X, shape=(N,1)),coefs)
        Y = Y[:,0] + intercepts[0]
        # no random component
        
        # include intercept
        linreg = LinearRegression(True)

        # test train function
        linreg.train(X,Y)
        for i in range(coefs.shape[0]):
            self.assertAlmostEqual(
                coefs[i,0],linreg.B[i],
                msg="Model coefficients did not agree.")
        for i in range(intercepts.size):
            self.assertAlmostEqual(
                intercepts[i],linreg.intercepts[i],
                msg="Model intercepts did not agree.")
        for i in range(len(linreg.r_squared)):
            self.assertAlmostEqual(
                1.0,linreg.r_squared[i],
                msg="Linear regression fit statistics inaccurate.")
        
        # test predict function
        Y_predict = linreg.predict(X)
        for i in range(Y.shape[0]):
            self.assertAlmostEqual(
                Y[i],Y_predict[i],
                msg="Predicted values did not agree.")
    
    def test_multiple_predictors_single_response(self):
        N = 10
        m = 2
        coefs = np.array([[5.0],[7.0]])
        X = np.random.rand(N,m)*10.0 
        Y = np.dot(X,coefs)
        Y = Y[:,0]
        # no random component
        
        # do not include intercept
        linreg = LinearRegression(False)

        # test train function
        linreg.train(X,Y)
        for i in range(coefs.shape[0]):
            self.assertAlmostEqual(
                coefs[i,0],linreg.B[i],
                msg="Model coefficients did not agree.")
        for i in range(len(linreg.r_squared)):
            self.assertAlmostEqual(
                1.0,linreg.r_squared[i],
                msg="Linear regression fit statistics inaccurate.")
        
        # test predict function
        Y_predict = linreg.predict(X)
        for i in range(Y.shape[0]):
            self.assertAlmostEqual(
                Y[i],Y_predict[i],
                msg="Predicted values did not agree.")
    
    def test_multiple_predictors_single_response_intercept(self):
        N = 10
        m = 2
        coefs = np.array([[5.0],[7.0]])
        intercepts = np.array([1.0])
        X = np.random.rand(N,m)*10.0 
        Y = np.dot(X,coefs)
        Y = Y[:,0] + intercepts[0]
        # no random component
        
        # include intercept
        linreg = LinearRegression(True)

        # test train function
        linreg.train(X,Y)
        for i in range(coefs.shape[0]):
            self.assertAlmostEqual(
                coefs[i,0],linreg.B[i],
                msg="Model coefficients did not agree.")
        for i in range(intercepts.size):
            self.assertAlmostEqual(
                intercepts[i],linreg.intercepts[i],
                msg="Model intercepts did not agree.")
        for i in range(len(linreg.r_squared)):
            self.assertAlmostEqual(
                1.0,linreg.r_squared[i],
                msg="Linear regression fit statistics inaccurate.")
        
        # test predict function
        Y_predict = linreg.predict(X)
        for i in range(Y.shape[0]):
            self.assertAlmostEqual(
                Y[i],Y_predict[i],
                msg="Predicted values did not agree.")
    
    def test_single_predictor_multiple_responses(self):
        N = 10
        coefs = np.array([[5.0,6.0]])
        X = np.random.rand(N)*10.0 
        Y = np.dot(np.ndarray(buffer=X, shape=(N,1)),coefs)
        # no random component

        # do not include intercept
        linreg = LinearRegression(False)

        # test train function
        linreg.train(X,Y)
        for i in range(coefs.shape[0]):
            for j in range(coefs.shape[1]):
                self.assertAlmostEqual(
                    coefs[i,j],linreg.B[i,j],
                    msg="Model coefficients did not agree.")
        for i in range(len(linreg.r_squared)):
            self.assertAlmostEqual(
                1.0,linreg.r_squared[i],
                msg="Linear regression fit statistics inaccurate.")
        
        # test predict function
        Y_predict = linreg.predict(X)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                self.assertAlmostEqual(
                    Y[i,j],Y_predict[i,j],
                    msg="Predicted values did not agree.")
    
    def test_single_predictor_multiple_responses_intercept(self):
        N = 10
        coefs = np.array([[5.0,6.0]])
        intercepts = np.array([1.0,2.0])
        X = np.random.rand(N)*10.0 
        Y = np.dot(np.ndarray(buffer=X, shape=(N,1)),coefs)
        for i in range(Y.shape[1]):
            Y[:,i] = Y[:,i] + intercepts[i]
        # no random component

        # include intercept
        linreg = LinearRegression(True)

        # test train function
        linreg.train(X,Y)
        for i in range(coefs.shape[0]):
            for j in range(coefs.shape[1]):
                self.assertAlmostEqual(
                    coefs[i,j],linreg.B[i,j],
                    msg="Model coefficients did not agree.")
        for i in range(intercepts.size):
            self.assertAlmostEqual(
                intercepts[i],linreg.intercepts[i],
                msg="Model intercepts did not agree.")
        for i in range(len(linreg.r_squared)):
            self.assertAlmostEqual(
                1.0,linreg.r_squared[i],
                msg="Linear regression fit statistics inaccurate.")
        
        # test predict function
        Y_predict = linreg.predict(X)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                self.assertAlmostEqual(
                    Y[i,j],Y_predict[i,j],
                    msg="Predicted values did not agree.")
    
    def test_multiple_predictors_multiple_responses(self):
        N = 10
        m = 2
        coefs = np.array([[5.0,6.0],[7.0,8.0]])
        X = np.random.rand(N,m)*10.0 
        Y = np.dot(X,coefs)
        # no random component

        # do not include intercept
        linreg = LinearRegression(False)

        # test train function
        linreg.train(X,Y)
        for i in range(coefs.shape[0]):
            for j in range(coefs.shape[1]):
                self.assertAlmostEqual(
                    coefs[i,j],linreg.B[i,j],
                    msg="Model coefficients did not agree.")
        for i in range(len(linreg.r_squared)):
            self.assertAlmostEqual(
                1.0,linreg.r_squared[i],
                msg="Linear regression fit statistics inaccurate.")
        
        # test predict function
        Y_predict = linreg.predict(X)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                self.assertAlmostEqual(
                    Y[i,j],Y_predict[i,j],
                    msg="Predicted values did not agree.")
    
    def test_multiple_predictors_multiple_responses_intercept(self):
        N = 10
        m = 2
        coefs = np.array([[5.0,6.0],[7.0,8.0]])
        intercepts = np.array([1.0,2.0])
        X = np.random.rand(N,m)*10.0 
        Y = np.dot(X,coefs)
        for i in range(Y.shape[1]):
            Y[:,i] = Y[:,i] + intercepts[i]
        # no random component

        # include intercept
        linreg = LinearRegression(True)

        # test train function
        linreg.train(X,Y)
        for i in range(coefs.shape[0]):
            for j in range(coefs.shape[1]):
                self.assertAlmostEqual(
                    coefs[i,j],linreg.B[i,j],
                    msg="Model coefficients did not agree.")
        for i in range(intercepts.size):
            self.assertAlmostEqual(
                intercepts[i],linreg.intercepts[i],
                msg="Model intercepts did not agree.")
        for i in range(len(linreg.r_squared)):
            self.assertAlmostEqual(
                1.0,linreg.r_squared[i],
                msg="Linear regression fit statistics inaccurate.")
        
        # test predict function
        Y_predict = linreg.predict(X)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                self.assertAlmostEqual(
                    Y[i,j],Y_predict[i,j],
                    msg="Predicted values did not agree.")


