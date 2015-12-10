import numpy as np

class LinearRegression:
   
    def __init__(self, include_intercept=True):
        self.include_intercept=include_intercept
        self.r_squared = []

    def train(self, X_train, Y_train, compute_stats=False):
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
        if compute_stats:
            # we store stats as members if we make them here
            # statistics will be in vectors p-long 
            # (the number of different quantities predicted)
            
            # get our least squares predictions
            Y_hat = np.dot(X,least_sq)
            
            self.r_squared = []
            for i in range(p):
                residual = Y[:,i] - Y_hat[:,i]
                ss_residual = np.dot(residual,residual)
                mean = np.sum(Y[:,i])/float(N)
                centered = Y[:,i] - mean
                ss_tot = np.dot(centered,centered)
                self.r_squared.append(1.0 - ss_residual/ss_tot)

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
        
        Y_predict = np.dot(X_predict,self.B)
        
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

