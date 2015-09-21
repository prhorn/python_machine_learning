import numpy as np
import sys
from scipy import stats

class decision_tree:
#{
   #SET DURING INIT:
   #   p = number of predictors, columns of X
   #   m = length of output vector, columns of Y
   #   n = number of training datapoints
   #   X = training inputs, do we really need to keep these? 
   #   Y = training outputs, do we really need to keep these? 
   #   is_classification = bool whether we have interger Y's
   #           if we are a classification problem we need m = 1
   #   base_node = the base tree_node object
   #   terminal_nodes = a list of the current terminal nodes (tree_node objects)
   #   n_obs_term_node =  a list of the number of observations for each terminal node  
   
   def __init__(self,X_,Y_,is_classification_):
   #{
      self.is_classification = is_classification_
      
      if (len(X_.shape)>1):
         self.X = np.array(X_)
      else:
         self.X = np.ndarray(buffer=X_,shape=(X_.size,1),dtype=np.float)
      self.p = self.X.shape[1]
      self.n = self.X.shape[0]
      
      if (len(Y_.shape)>1):
         self.Y = np.array(Y_)
      else:
         if is_classification:
            self.Y = np.ndarray(buffer=Y_,shape=(Y_.size,1),dtype=np.int)
         else:
            self.Y = np.ndarray(buffer=Y_,shape=(Y_.size,1),dtype=np.float)
      self.m = self.Y.shape[1]
      if not (self.n == self.Y.shape[0]):
         print 'passed training data matrices incommensurate'
         sys.exit(1)
      
      if (self.is_classification)  and (not (self.m == 1)):
         print 'classification only implemented for a single set of classes labeled (0-n_class-1)'
         sys.exit(1)
      
      #make the base node
      if self.is_classification:
         base_value = stats.mode(self.Y[:,0])[0][0]
      else:
         base_value = np.mean(self.Y,axis=0)  
      self.base_node = tree_node(base_value,self.X,self.Y,self.is_classification,None)
      self.terminal_nodes = []
      self.terminal_nodes.append(self.base_node)
      self.n_obs_term_node = []
      self.n_obs_term_node.append(self.base_node.obs_for_node()) 
   #}
   
   def add_a_branch(self):
   #{
      #add a branch by recursive bianry splitting
      #for all terminal branches, determine the optimal cut
      success = False #we will return true if we add a branch
      best_node_index = -1 #the node that will get the cut
      best_node_error_change = -1.0E-10 #I don't like the idea of comparing zeros
      best_node_cut_index = -1 #the column of X for the cut
      best_node_cut_value = -1.0 
      for n in range(len(self.terminal_nodes)):
         valid_n,error_change_n,cut_index_n,cut_value_n = self.terminal_nodes[n].determine_optimal_cut() 
         if valid_n and (error_change_n < best_node_error_change):
            best_node_index = n
            best_node_error_change = error_change_n
            best_node_cut_index = cut_index_n
            best_node_cut_value = cut_value_n

      #perform the cut and update list of terminal branches and their ranks
      if (best_node_index >= 0):
         #we found a node to cut that reduced error
         #remove this as a terminal node
         best_node = self.terminal_nodes.pop(best_node_index)
         best_node_size = self.n_obs_term_node.pop(best_node_index)
         #make the branches
         best_node.make_branches(best_node_cut_index,best_node_cut_value)
         #update the list of terminal nodes
         self.terminal_nodes.append(best_node.branches[0])
         self.terminal_nodes.append(best_node.branches[1])
         self.n_obs_term_node.append(best_node.branches[0].obs_for_node())
         self.n_obs_term_node.append(best_node.branches[1].obs_for_node())
         success = True
      #convey that we added a branch
      return success
   #}

   def predict(self,X_predict):  
   #{
      if (len(X_predict.shape) == 1):
         n_predictions = X_predict.size
      else:
         n_predictions = X_predict.shape[0]
      
      if self.is_classification:
         Y_predict = np.ndarray(shape=(n_predictions,self.m),dtype=np.int)
      else:
         Y_predict = np.ndarray(shape=(n_predictions,self.m),dtype=np.float)

      for i in range(n_predictions):
         Y_predict[i,:] = self.base_node.predict(X_predict[i,:])
      
      return Y_predict
   #}
#}

class tree_node:
#{
   #is_terminal:  True if this is a terminal node
   #value:        The expected value for observations ending at this node
   #parent_node:  the parent tree_node for which we are a branch
   #cut_index:    The index of the predictor vector element that we use to split
   #              -1 until not is_terminal
   #cut_value:    The value that determines which branch from this node we follow
   #              0.0 until not is_terminal
   #branches:     list of [less_than,greater_than] branches from this node
   #              [] until not is_terminal
   
   #X_train,Y_train: training data that meets the requirements for this node
   #is_classification: if true Y has ints
   
   #determined_opt_cut: bool saying whether we have already computed the optimal
   #                    next branching from this node
   #opt_cut_valid
   #opt_cut_index
   #opt_cut_value
   #opt_error_change
   def __init__(self,value_,X_train_,Y_train_,is_classification_,parent_node_=None):
   #{
      self.value = value_
      self.is_terminal = True
      self.cut_index = -1
      self.cut_value = 0.0
      self.branches = []
      self.parent_node = parent_node_
      self.X_train = np.array(X_train_) #make sure we have our own copy
      self.Y_train = np.array(Y_train_) 
      self.is_classification = is_classification_
      self.determined_opt_cut = False
   #}

   def make_branches(self,cut_index_,cut_value_):
   #{
      self.cut_index = cut_index_
      self.cut_value = cut_value_
      
      #order our data so that dividing our training data between the branches is easier
      sort_indices = np.argsort(self.X_train[:,self.cut_index]) #ascending
      self.X_train = self.X_train[sort_indices,:]
      self.Y_train = self.Y_train[sort_indices,:]
      
      #determine where our data is divided
      first_greater_eq_index = -1
      for r in range(self.X_train.shape[0]):
         if (self.X_train[r,self.cut_index] >= self.cut_value):
            first_greater_eq_index = r
            break
      
      #zero dimension check
      if (first_greater_eq_index <1):
         print 'our partitioning of this node resulted in a branch having zero obs'
         sys.exit(1)

      #find the predicted values of Y for the branches
      if self.is_classification:
         less_than_value = stats.mode(self.Y_train[:first_greater_eq_index,0])[0][0]
         greater_than_value = stats.mode(self.Y_train[first_greater_eq_index:,0])[0][0]
      else:
         less_than_value = np.mean(self.Y_train[:first_greater_eq_index,:],axis=0)  
         greater_than_value = np.mean(self.Y_train[first_greater_eq_index:,:],axis=0)  
      
      #construct the new branches
      if not (len(self.branches)==0):
         self.branches = [] #I don't think we will want to call make branches twice, but on the off chance
      
      self.branches.append(tree_node(less_than_value,self.X_train[:first_greater_eq_index,:],self.Y_train[:first_greater_eq_index,:],self.is_classification,self))
      self.branches.append(tree_node(greater_than_value,self.X_train[first_greater_eq_index:,:],self.Y_train[first_greater_eq_index:,:],self.is_classification,self))
      self.is_terminal = False
   #}
   
   def compute_error(self,temp_Y,use_classification_error_rate=True):
   #{
      #if use_classification_error_rate is true, use the 
      #classification error rate instead of the gini index for class trees
      #compute our error measure for a set of Ys
      #we can also compute for out Ys or a subset of our Ys for instance
      #error will be a scalar even if we are predicting multiple outputs
      error = 0.0
      if self.is_classification:
         if use_classification_error_rate:
            #our error is the classification error rate
            uniques,counts = np.unique(temp_Y[:,0],return_counts=True) 
            fracs_group = counts / float(temp_Y.shape[0])
            error = 1.0 - max(fracs_group)
         else:
            #gini index - measure of node purity
            #we need to determine the fraction of our training observations coming from each class
            uniques,counts = np.unique(temp_Y[:,0],return_counts=True) 
            fracs_group = counts / float(temp_Y.shape[0])
            for i in fracs_group:
               error += i * (1.0 - i)
      else:
         temp_value = np.mean(temp_Y,axis=0)
         if (temp_Y.shape[1] == 1): #special case because value is not a vector
            error = sum( (temp_Y[:,0] - temp_value)**2 ) 
         else:
            for i in range(temp_Y.shape[1]):
               error += sum( (temp_Y[:,i] - temp_value[i])**2 )
      
      return error
   #}

   def determine_optimal_cut(self):
   #{
      
      if not self.determined_opt_cut: #internal variable telling us if we need to do this calcualtion
         parent_node_error = self.compute_error(self.Y_train)
         
         self.opt_cut_valid = False #set to true below if we find a valid cut 
         self.opt_error_change = -1.0E-10 #make sure we don't cut if we can't reduce the residual 
         self.opt_cut_index = -1
         self.opt_cut_value = -1.0
         #if I understand the algorithm correctly we
         #evaluate error change for all possible
         #cut indices and values -- yuck
         for c in range(self.X_train.shape[1]):
            sort_indices = np.argsort(self.X_train[:,c])
            x_col_sorted = self.X_train[sort_indices,c]
            y_sorted     = self.Y_train[sort_indices,:]
            #for this c compute the best division
            for start_of_greater in range(1,x_col_sorted.size):
               cur_error = self.compute_error(y_sorted[:start_of_greater,:]) + self.compute_error(y_sorted[start_of_greater:,:])  
               cur_error_change = cur_error - parent_node_error
               if (cur_error_change < self.opt_error_change):
                  #we have found a better split
                  self.opt_cut_valid = True
                  self.opt_cut_index = c
                  self.opt_error_change = cur_error_change
                  #average of the first x in the greater block and last in the less block
                  self.opt_cut_value = x_col_sorted[start_of_greater-1] + (x_col_sorted[start_of_greater] - x_col_sorted[start_of_greater-1])/2.0
         self.determined_opt_cut = True
      #else:
         #we have already done this calculation and stored the results as members

      return self.opt_cut_valid,self.opt_error_change,self.opt_cut_index,self.opt_cut_value
   #}

   def obs_for_node(self):
   #{
      return self.X_train.shape[0]
   #}

   def predict(self,x):
   #{
      if self.is_terminal:
         return self.value
      else:
         if (x[self.cut_index] < self.cut_value):
            return self.branches[0].predict(x)
         else:
            return self.branches[1].predict(x)
   #}
   
#}

