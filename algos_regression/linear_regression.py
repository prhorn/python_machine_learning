import math

import numpy as np
import unittest
import scipy.stats

class LinearRegression:
   
    def __init__(self, is_weighted=False, include_intercept=True, compute_stats=True, training_confidence_level=95):
        self.is_weighted = is_weighted
        self.include_intercept = include_intercept
        self.compute_stats = compute_stats
        self.training_confidence_level = training_confidence_level
        
        self.stats_discriptions = {
            'r_squared':'coefficient of determination',
            'f_stat_mean':'F-statistic for comparison to mean (intercept only) model',
            'p_value_mean':'p-value for comparision to mean (intercept only) model',
            'parameter_stats':'for each coef (and intercept): [param_value t-value p-value lower_bound upper_bound]'}

        
    def train(self, X_train, Y_train, W_train=None):
        # input:
        #  X_train -- Nxm feature matrix or vector if m==1
        #  Y_train -- Nxp target matrix or vector if p==1
        #
        # output:
        #  self.B  -- mxp coefficient matrix or vector if p==1
        #  self.intercepts --   p-long vector of intercepts if
        #                       (self.include_intercept)
        #  other members for computing confidence intervals 
        
        # sort our dimensions of inputs
        self.N = X_train.shape[0]
        if self.N != Y_train.shape[0]:
            raise ValueError("Input training X and Y did not have the "
                             "same number of observations")
        
        # we will accept W as a full square matrix (assumed symmetric) or as a vector
        if self.is_weighted:
            # test if we have passed something plausubly a set of weights
            # if we are doing weighted regression
            try:
                w_rows = W_train.shape[0]
            except:
                raise ValueError("Input weights not expected type")
            
            # see if dimensions are consistent with number of observations
            # in other passed quantities
            if w_rows != self.N:
                raise ValueError("Input training X and W did not have the "
                                 "same number of observations")
                                 
            # store passed W in a consistent 2-D form
            try:
                w_cols = W_train.shape[1]
            except:
                # passed W is a 1-D array
                self.W = np.diag(W_train)
            else:
                # passed W is a 2-D array, but is it a column vec?
                if w_cols == 1:
                    # a column vector
                    self.W = np.diag(W_train[:,0])
                elif w_rows == 1:
                    # a row vector. strange, but whatever
                    self.W = np.diag(W_train[0,:])
                else:
                    # passed W is a matrix, but is it square?
                    if w_cols != w_rows:
                        raise ValueError("Input weight matrix was not square")
                    self.W = np.array(W_train)
        
        # determine the number of features (X cols)
        try:
            self.m = X_train.shape[1]
        except:
            self.m = 1
        
        # determine the number of quantities predicted
        try:
            self.p = Y_train.shape[1]
        except:
            self.p = 1
        
        # If we are including an intecept, then we need to augment
        # the passed X_train with a column of 1's as X
        # X will be a matrix after this step either way
        if self.include_intercept:
            self.X = np.column_stack((X_train,np.ones(self.N)))
        else:
            self.X = np.ndarray(buffer=X_train, shape=(self.N,self.m), dtype=np.float)
        
        # make Y a matrix as well so that B is consistently a matrix
        self.Y = np.ndarray(buffer=Y_train, shape=(self.N,self.p), dtype=np.float)
        
        # determine coefficients and intercepts
        # store these intermeidates as memeber to compute confidence intervals later
        if self.is_weighted:
            # weighted
            # b = (X.T W X)^-1 (X.T W Y)
            self.metric = np.dot(self.X.T,np.dot(self.W,self.X))
            self.metric_inv = np.linalg.pinv(self.metric)
            self.least_sq = np.dot(self.metric_inv,np.dot(self.X.T,np.dot(self.W,self.Y)))
        else:
            # unweighted
            # b = (X.T X)^-1 (X.T Y)
            self.metric = np.dot(self.X.T,self.X)
            self.metric_inv = np.linalg.pinv(self.metric)
            self.least_sq = np.dot(self.metric_inv,np.dot(self.X.T,self.Y))
        
        # store the least squares results
        if self.include_intercept:
            self.intercepts = np.array(self.least_sq[-1,:])
            if self.p==1:
                self.B = np.array(self.least_sq[:-1,0])
            else:
                self.B = np.array(self.least_sq[:-1,:])
        else:
            if self.p==1:
                self.B = np.array(self.least_sq[:,0])
            else:
                self.B = np.array(self.least_sq)
        
        # compute statistics about the fit if asked
        if self.compute_stats:
            if not self.is_weighted:
                self.compute_fit_statistics()
            # else NYI for weighted TODO
        # done training. all results are stored as members

    def compute_fit_statistics(self):
        if not hasattr(self, 'B'): 
            raise RuntimeError("Model must be trained before computing fit statistics")
        
        # we store stats as members if we make them here
        # statistics will be in vectors p-long 
        # (the number of different quantities predicted, cols of Y)
        # even if p == 1

        # get our least squares predictions
        Y_hat = np.dot(self.X,self.least_sq)
        
        # determine the threshold t-value for our coefficient confidence intervals
        alpha = 1.0 - float(self.training_confidence_level)/100.0
        t_dof = self.N - self.least_sq.shape[0]
        thresh_t_value = scipy.stats.t.isf(alpha/2.0, t_dof) 
        
        self.r_squared = []
        self.f_stat_mean = []
        self.p_value_mean = []
        self.parameter_stats = []
        for i in range(self.p):
            residual = self.Y[:,i] - Y_hat[:,i]
            ss_residual = np.dot(residual,residual)
            mean = np.sum(self.Y[:,i])/float(self.N)
            centered = self.Y[:,i] - mean
            # equivalent to the ss_residual for the mean model (intercept only)
            ss_tot = np.dot(centered,centered)
            # the coefficient of determination, r_squared
            self.r_squared.append(1.0 - ss_residual/ss_tot)
            
            # variance of the residuals (denominator accounts for intercept)
            residual_var = ss_residual/float(t_dof)
            # variances of the linear coefficients and intercepts
            parameter_vars = self.metric_inv.diagonal()*residual_var  
            
            # compute f-statistic for the mean model hypothesis
            self.f_stat_mean.append((ss_tot - ss_residual)/
                                    float(self.least_sq.shape[0] - 1)/residual_var)
            # the corresponding p-value for the mean model hypothesis
            self.p_value_mean.append(
                scipy.stats.f.sf(self.f_stat_mean[i], self.least_sq.shape[0] - 1, t_dof))
            
            # statistics on our coefficients (and intercepts) will be collected as follows:
            # param_value t-value p-value lower_bound upper_bound
            stats_i = np.empty((self.least_sq.shape[0],5), dtype=np.float)
            # loop over parameters
            for j in range(self.least_sq.shape[0]): 
                # parameter value
                stats_i[j,0] = self.least_sq[j,i]
                # t-value
                stats_i[j,1] = self.least_sq[j,i]/math.sqrt(parameter_vars[j])
                # p-value
                stats_i[j,2] = 2.0*scipy.stats.t.sf(stats_i[j,1],t_dof)
                # confidence interval lower bound
                stats_i[j,3] = self.least_sq[j,i] - thresh_t_value*math.sqrt(parameter_vars[j])
                # confidence interval upper bound
                stats_i[j,4] = self.least_sq[j,i] + thresh_t_value*math.sqrt(parameter_vars[j])
            self.parameter_stats.append(stats_i)
    
    def predict(self,X_predict,confidence_level=95.0):
        # see if we have trained
        if not hasattr(self, 'B'): 
            raise RuntimeError("Model must be trained before predicting")
        
        # if X_predict is a vector, we will check the dimensions of B
        # to determine whether it is a row (observation) or a column
        # (multiple observations one feature). If it is 1x1, it doesn't matter
        if len(X_predict.shape) == 1:
            if self.B.shape[0] == 1:
                # multiple observations
                X_pred = np.array(X_predict.reshape(X_predict.size,1)) 
            else: 
                # a single observation
                X_pred = np.array(X_predict.reshape(1,X_predict.size)) 
        else:
            X_pred = np.array(X_predict)
        
        N_pred = X_pred.shape[0]
        m_pred = X_pred.shape[1]

        # see if these dimensions are commensurate with
        # those used to train our coefficents
        if m_pred != self.B.shape[0]:
            raise ValueError("Input X for prediction did not have " 
                             "the expected number of columns")
        
        if self.include_intercept:
            X_aug = np.column_stack((X_pred,np.ones(N_pred)))
        else:
            X_aug = np.array(X_pred)
        
        # Y_pred will always be an N_pred by p matrix
        Y_pred = np.dot(X_aug,self.least_sq)
        
        # compute prediction intervals if requested
        if self.compute_stats:
            if self.p != 1:
                raise RuntimeError("prediction intervals NYI for more than 1 col in Y")

            # estimate residual degrees of freedom
            #eye_min_H = np.eye(self.N) - np.dot(np.dot(self.X,np.dot(self.metric_inv,self.X.T)),self.W)
            #tmp_res_dof = np.trace(np.dot(eye_min_H.T,eye_min_H))
            # R does it the stupid way, so we will too
            res_dof = self.N - self.least_sq.shape[0] 
            #print 'modified residual DOF ',tmp_res_dof
            #print 'normal residual DOF ',res_dof
            
            # weights that are zero within thresh should not be classified to N
            if self.is_weighted:
                w_diag = np.diag(self.W)
                weight_thresh = 1.0E-6
                for w in w_diag:
                    if w < weight_thresh:
                        res_dof = res_dof - 1
            
            #print 'threshed residual DOF ',res_dof

            # determine the threshold t-value for our coefficient confidence intervals
            alpha = 1.0 - float(confidence_level)/100.0
            thresh_t_value = scipy.stats.t.isf(alpha/2.0, res_dof) 
            
            fitting_residuals = self.Y - np.dot(self.X,self.least_sq)
            if self.is_weighted:
                resid_var = np.trace(np.dot(fitting_residuals.T,np.dot(self.W,fitting_residuals)))
                resid_var = resid_var/float(res_dof)
            else:
                resid_var = np.trace(np.dot(fitting_residuals.T,fitting_residuals))
                resid_var = resid_var/float(res_dof)
            
            self.prediction_interval_halfwidths = np.empty(N_pred, dtype=np.float)
            for i in range(N_pred):
                var_mean = np.dot(X_aug[i,:],np.dot(self.metric_inv,X_aug[i,:]))*resid_var + resid_var
                self.prediction_interval_halfwidths[i] = thresh_t_value*math.sqrt(var_mean)
        # done with prediction intervals

        # Y_pred is N_pred by p, but we will return 
        # scalar, mat, or vec as appropriate
        ###
        # if X_predict was a single obs and p == 1, we should return a scalar
        # if X_predict was a single obs and p > 1, we should return a vector
        # if X_predict was a matrix (many obs) and p == 1, we should return a vector
        # if X_predict was a matrix (many obs) and p > 1, we should return a matrix
        if N_pred == 1 and self.p == 1:
            Y_predict = Y_pred[0,0]
        elif N_pred == 1 and self.p > 1:
            Y_predict = np.array(Y_pred[0,:])
        elif self.p == 1:
            Y_predict = np.array(Y_pred[:,0])
        else:
            Y_predict = np.array(Y_pred)
        
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


