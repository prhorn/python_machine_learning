import copy
import math
import sys
import os

import numpy as np

sys.path.append(os.getcwd()+'/nonlinear_solvers/')
from l_bfgs import l_bfgs
from steepest_descent import steepest_descent

def sigmoid(t):
    if (t < -700.0):
        result = 0.0
    else:
        result = (1.0/(1.0 + math.exp(-1.0*float(t))))
    return result

def sigmoid_deriv(t):
    s = sigmoid(t)
    return s*(1.0-s)

def elementwise_function_on_vector(T,func):
    result = np.empty(T.size);
    for i in range(T.size):
        result[i] = func(T[i])
    return result

def elementwise_function_on_matrix(T,func):
    result = np.empty(T.shape);
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            result[i,j] = func(T[i,j])
    return result

def softmax_on_vector(T):
    exp_T = elementwise_function_on_vector(T,math.exp)
    denominator = np.sum(exp_T)
    return (1.0/denominator) * exp_T


class NeuralNetwork:

    def __init__(self, n_base_features, n_units_per_layer, regularization_parameter=0.0, is_classification=False, n_outputs=1):
        # TODO add valid parameter tests
        # regularization_parameter >= 0; 0 yields unregularized
        self.regularization_parameter = regularization_parameter
        self.is_classification = is_classification
        # n_outputs >= 1; 1 for regression. more for classification
        self.n_outputs = n_outputs 
        self.n_base_features = n_base_features
        self.n_units_per_layer = copy.copy(n_units_per_layer)
        self.n_hidden_layers = len(self.n_units_per_layer) 
        # number of layers + 1 coefficient matrices
        # the first matrix [0] is n_base_features+1(intercept) by n_units_per_layer[0]
        # the ith matrix is n_units_per_layer[i-1]+1 by n_units_per_layer[i] for i (1,layers-1) inclusive
        # the last matrix [layers] is n_units_per_layer[layers-1]+1 by n_outputs
        self.layer_coefficients = []
        self.layer_coefficients_origin = []
        self.obtain_guess()

    def obtain_guess(self):
        # ESL suggets random but near zero params
        # random, uniform [-0.7,0.7] for standardized inputs
        # base features to first hidden layer
        self.layer_coefficients.append(
            np.random.rand(self.n_base_features + 1, self.n_units_per_layer[0])*0.8 - 0.4)
        # hidden layer to hidden layer
        for i in range(1,self.n_hidden_layers):
            self.layer_coefficients.append(
                np.random.rand(self.n_units_per_layer[i-1] + 1, self.n_units_per_layer[i])*0.8 - 0.4)
        # hidden layer to output
        self.layer_coefficients.append(
            np.random.rand(self.n_units_per_layer[self.n_hidden_layers-1] + 1, self.n_outputs)*0.8 - 0.4)
        self.ls_origin_to_current_pos()
        self.gradient_length = 0
        for i in range(len(self.layer_coefficients)):
            self.gradient_length = self.gradient_length + self.layer_coefficients[i].size

    def predict(self,X_predict):
        # base features to first hidden layer (adding intercept 'feature')
        self.layer_features = []
        # X itself is the zeroth element of the features vector
        self.layer_features.append(X_predict)
        # layer_features = [X, actual_hidden_layers, T]
        augmented_X = np.column_stack((X_predict,np.ones(X_predict.shape[0])))
        current_features = np.dot(augmented_X,self.layer_coefficients[0])
        # hidden layer (to hidden layers) to untransformed result
        for i in range(1,self.n_hidden_layers+1):
            current_features = elementwise_function_on_matrix(current_features,sigmoid)
            # add the features for the current hidden layer
            self.layer_features.append(current_features)
            augmented_current_features = np.column_stack((current_features,np.ones(current_features.shape[0])))
            current_features = np.dot(augmented_current_features,self.layer_coefficients[i])
        # collect the final 'features', T where Y_hat = g(T)
        self.layer_features.append(current_features)
        # final transformation, g()
        if self.is_classification:
            # softmax
            Y = np.empty(current_features.shape, dtype=np.float)
            for i in range(Y.shape[0]):
                Y[i,:] = softmax_on_vector(current_features[i,:])
        else:
            # identity
            Y = np.array(current_features)
        return Y 
    
    def value(self):
        # currently only squared error loss
        Y_predict = self.predict(self.X_train)
        # currently only squared error
        result = 0.0
        for i in range(self.Y_train.shape[0]):
            for j in range(self.Y_train.shape[1]):
                result = result + (Y_predict[i,j] - self.Y_train[i,j])**2
        # add regularization penalty
        penalty_term = 0.0
        for i in range(len(self.layer_coefficients)):
            for j in range(self.layer_coefficients[i].shape[0]-1): #ignore intercepts
                for k in range(self.layer_coefficients[i].shape[1]):
                    penalty_term = penalty_term + self.layer_coefficients[i][j,k]**2
        result = result + self.regularization_parameter*penalty_term
        return result

    def gradient(self):
        # first we need our residuals 
        Y_predict = self.predict(self.X_train)
        # \frac{\partial RSS}{\partial Y_hat} = -2R
        R = self.Y_train - Y_predict

        # Y_hat = g(T)
        # compute the derivitive of the RSS wrt T as R_hat
        # T is the last matrix in 
        if self.is_classification:
            # we have to chain through g = softmax
            # R_hat = term1 + term2 =
            # \frac{\partial RSS}{\partial T_{ab}} = 
            #     -2 R_{ab} exp(T_{ab}) * \frac{1}{\sum_c exp(T_{ac})}
            #     +2 exp(T_{ab}) * \frac{\sum_k R_{ak} exp(T_{ak})}{[\sum_c exp(T_{ac})]^2} 
            exp_T = np.zeros(self.layer_features[self.n_hidden_layers+1].shape)
            exp_T_row_sum = np.zeros(self.layer_features[self.n_hidden_layers+1].shape[0])
            for r in range(self.layer_features[self.n_hidden_layers+1].shape[0]):
                row = elementwise_function_on_vector(self.layer_features[self.n_hidden_layers+1][r,:],math.exp)
                exp_T[r,:] = row
                exp_T_row_sum[r] = np.sum(row)
            term1 = -2.0 * R * exp_T
            for r in range(term1.shape[0]):
                term1[r,:] = (1.0/exp_T_row_sum[r]) * term1[r,:]
            ReTt = np.dot(R,exp_T.T)
            term2 = 2.0 * exp_T
            for r in range(term2.shape[0]):
                term2[r,:] = (ReTt[r,r]/(exp_T_row_sum[r]**2)) * term2[r,:]
            R_hat = term1 + term2
        else:
            # for regression we just use the identity
            # R_hat = \frac{\partial RSS}{\partial T_{ab}} =  -2R_{ab}
            R_hat = -2.0*R
        # all difference between regression and classification is now in R_hat
        # now we need to consider derivatives of T
        # our function:
        # layer_coefficients = B
        # layer_features = [X, actual_hidden_layers, T] = F
        # F[i] = sigmoid[(F[i-1],1) B[i-1]]
        # F[i+1] = sigmoid[(F[i],1) B[i]]
        # T = F[layers+1] = F[layers] B[layers]
        # T = F[-1] = F[-2] B[-1]
        # The final layer is distinct from the others
        #
        # our derivatives:
        #
        # We need F derivatives to continue chaining
        # We only collect coefficient derivatives
        #
        # \frac{\partial RSS}{\partial B[layers]_{rs}} = \sum_a (F[layers],1)_{ar} (R_hat)_{as}
        #
        # \frac{\partial RSS}{\partial F[layers]_{rs}} = \sum_a (R_hat)_{ra} B[layers]_{sa} (not last row of B)
        #
        # for i <= layers:
        #
        # \frac{\partial F[i+1]_{cd}}{\partial F[i]_{rs}} = 
        #     sigmoid_prime[(F[i],1) B[i]]_{cd} \delta_{cr} B[i]_{sd} (not last row of B)
        #
        # \frac{\partial F[i+1]_{cd}}{\partial B[i]_{rs}} = 
        #     sigmoid_prime[(F[i],1) B[i]]_{cd} \delta_{ds} (F[i],1)_{cr}
        # 
        #collect matrices of coef derivatives here for now. vectorize later
        grad_list = [0]*len(self.layer_coefficients)
        # we start with the last set of features and parameters,
        # those that yield T
        # form (F[layers],1)
        F_aug = np.column_stack((self.layer_features[self.n_hidden_layers],
                                 np.ones(self.layer_features[self.n_hidden_layers].shape[0])))
        current_B_deriv = np.dot(F_aug.T,R_hat)
        current_F_deriv = np.dot(R_hat,np.transpose(self.layer_coefficients[self.n_hidden_layers][:-1,:])) 
        current_penalty_grad = self.regularization_parameter*2.0*self.layer_coefficients[self.n_hidden_layers][:-1,:]
        grad_list[self.n_hidden_layers] = current_B_deriv
        grad_list[self.n_hidden_layers][:-1,:] = grad_list[self.n_hidden_layers][:-1,:] + current_penalty_grad
        for i in reversed(range(self.n_hidden_layers)):
            # first, lets get sigmoid_prime[(F[i],1) B[i]]
            F_aug = np.column_stack((self.layer_features[i],np.ones(self.layer_features[i].shape[0])))
            sig_prime = elementwise_function_on_matrix(np.dot(F_aug,self.layer_coefficients[i]) ,sigmoid_deriv)
            # chain current B through last F deriv then update last F deriv to current
            current_B_deriv = np.dot(F_aug.T,sig_prime * current_F_deriv)
            current_F_deriv = np.dot(current_F_deriv * sig_prime,np.transpose(self.layer_coefficients[i][:-1,:]))
            current_penalty_grad = self.regularization_parameter*2.0*self.layer_coefficients[i][:-1,:]
            grad_list[i] = current_B_deriv
            grad_list[i][:-1,:] = grad_list[i][:-1,:] + current_penalty_grad
        # now we need to construct the vectorized gradient
        grad = np.zeros(self.gradient_length, dtype=np.float)
        grad_off = 0
        for i in range(len(self.layer_coefficients)):
            grad[grad_off : grad_off + grad_list[i].size] = np.reshape(grad_list[i], grad_list[i].size, order='F') 
            grad_off = grad_off + grad_list[i].size
        
        # FD test says divide by 2
        grad = 0.5*grad
        # FD test 
        # print 'begin FD test of gradient'
        # fd_grad = self.finite_difference_gradient()
        # grad_diff = grad - fd_grad
        # print 'square length of analytic: ',np.dot(grad,grad)
        # print 'square length of fd      : ',np.dot(fd_grad,fd_grad)
        # print 'sum sq analytic min fd   : ',np.dot(grad_diff,grad_diff)
        # print 'inner prod of A and FD   : ',np.dot(grad,fd_grad)
        # print 'analytic grad: '
        # print grad
        # print 'fd grad: '
        # print fd_grad
        return grad

    def finite_difference_gradient(self, step_size=1.0E-4):
        # store our current position as the finite difference 'origin;
        self.fd_origin_to_current_pos()
        # the gradient we will fill
        grad = np.zeros(self.gradient_length)
        for i in range(self.gradient_length):
            disp = np.zeros(self.gradient_length)
            # forward step along component i
            disp[i] = step_size
            self.update(disp)
            v_forward = self.value()
            # return to fd origin
            self.move_to_fd_origin()
            # back step along component i
            disp[i] = -1.0*step_size
            self.update(disp)
            v_back = self.value()
            # return to fd origin
            self.move_to_fd_origin()
            #compute gradient for this parameter 
            grad[i] = (v_forward - v_back)/(2.0*step_size)
        return grad 

    def update(self, disp):
        disp_offset = 0
        for i in range(len(self.layer_coefficients)):
            disp_for_i = disp[disp_offset : disp_offset + self.layer_coefficients[i].size]
            #column major. consistent with grad 
            update_for_i = np.reshape(disp_for_i, self.layer_coefficients[i].shape, order='F') 
            self.layer_coefficients[i] = self.layer_coefficients[i] + update_for_i
            disp_offset = disp_offset + self.layer_coefficients[i].size

    def ls_origin_to_current_pos(self):
        self.layer_coefficents_origin = []
        for i in range(len(self.layer_coefficients)):
            self.layer_coefficents_origin.append(np.array(self.layer_coefficients[i]))
   
    def move_to_ls_origin(self):
        self.layer_coefficents = []
        for i in range(len(self.layer_coefficients_origin)):
            self.layer_coefficents.append(np.array(self.layer_coefficients_origin[i]))
    
    def fd_origin_to_current_pos(self):
        self.layer_coefficients_fd = []
        for i in range(len(self.layer_coefficients)):
            self.layer_coefficients_fd.append(np.array(self.layer_coefficients[i]))
   
    def move_to_fd_origin(self):
        self.layer_coefficents = []
        for i in range(len(self.layer_coefficients_fd)):
            self.layer_coefficents.append(np.array(self.layer_coefficients_fd[i]))
   
    def train(self, X_train, Y_train, tol=1.0E-7, algo=1, print_iter=False):
        # TODO reexpression of class labels as vectors
        self.X_train = X_train
        if self.is_classification:
            # we assume we have been passed a vector of integer labels
            self.Y_train = np.zeros((Y_train.shape[0], np.amax(Y_train)+1), dtype=np.int)
            for i in range(Y_train.shape[0]):
                self.Y_train[i,Y_train[i,0]] = 1
        else:
            self.Y_train = Y_train
        self.tolerance = tol
        if algo == 0:
            optimizer = steepest_descent(self)
        elif algo == 1:
            optimizer = l_bfgs(self,20,0,0.5,0)
        else:
            print 'optimizer not recognized'
        max_iter = 5000
        converged = False
        cur_iter = 0
        print 'beginning optimization of neural network'
        for i in range(max_iter):
            cur_iter = cur_iter + 1
            converged = optimizer.next_step() 
            if (converged):
                break
            if (print_iter):
                print "  "+str(cur_iter)+"  "+"{:.12f}".format(optimizer.value)+"  "+\
                    str(optimizer.error)+"  "+optimizer.comment
        if (converged):
            print "  "+str(cur_iter)+"  "+"{:.12f}".format(optimizer.value)+"  "+\
                str(optimizer.error)+"  Optimization Converged"
        else:
            print "  "+str(cur_iter)+"  "+"{:.12f}".format(optimizer.value)+"  "+\
                str(optimizer.error)+"  Optimization Failed"
        return converged

