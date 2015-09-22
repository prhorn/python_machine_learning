import numpy as np
import sys
from scipy import stats
from copy import *

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
   
   def __deepcopy__(self,memo):
   #{
      #TODO I'm not really sure what I should be doing with memo...
      the_copy = decision_tree(np.array(self.X),np.array(self.Y),copy.deepcopy(self.is_classification))
      
      #we still need to handle:
      #   base_node 
      #   terminal_nodes 
      #   n_obs_term_node
      #copy the base node and all connected nodes
      #we made our own separate function for copy because we want to pass on parent info
      #the base node has no parent 
      the_copy.base_node = self.base_node.return_deepcopy(None) 
      
      #clear the existing terminal node lists
      the_copy.n_opbs_term_node = []
      the_copy.terminal_nodes = []
      #repopulate the lists
      the_copy.terminal_nodes = the_copy.base_node.return_list_terminal_nodes()
      for i in the_copy.terminal_nodes:
         the_copy.n_obs_term_node.append(i.obs_for_node())
      return the_copy
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
   
   def prune(self,alpha):
   #{
      #perform cost complexity pruning of the tree with pruning parameter alpha
      #this function modifies self to be the pruned tree
      if not alpha > 0.0:
         print 'pruning parameter cannot be negative. passed alpha='+str(alpha)
         sys.exit(1)
      
      #generated trees should be nested, so we will always consider popping from the terminal
      #node list of the next higher rank space
      #start with the node list of the max rank space
      best_terminal_node_list = copy(self.terminal_nodes) #still references same objects
      #get the error for the full tree. we will have to beat this to delete any branches
      best_prune_error = 0.0 
      for i in best_terminal_node_list:
         best_prune_error += (i.compute_error(i.Y_train,True) + alpha)
      
      #now we consider trees of reduced rank (number of terminal nodes)
      max_rank = len(self.terminal_nodes)
      for rank_reduction in range(1,max_rank): #consider removing 1 through all but 1. This is the max number of iter. we may break sooner
         #depending on the tree structure we could potentially remove multiple at a time, which makes this a bit messy
         #consider making the parent of each terminal node of the current best a terminal node itself, removing its progeny as terminal nodes
         if len(best_terminal_node_list) == 1:
            break #there are no more nodes to remove
         best_terminal_node_list_for_rank = []
         best_prune_error_for_rank = 999.9
         found_better_for_rank = False
         n_nodes_to_consider = len(best_terminal_node_list)
         for i in range(n_nodes_to_consider):
            nodes_i = copy(best_terminal_node_list)
            #remove this node
            node_removed = nodes_i.pop(i)
            if (node_removed.parent_node is None):
               continue #we can't remove the base node. go to the next
            #remove all other nodes related to the parent of this node
            for j in nodes_i:
               if not (node_removed.parent_node is None):
                  if node_removed.parent_node.is_progeny(j):
                     nodes_i.remove(j)
            #add the parent of the pop'd node as a terminal node itslef
            nodes_i.append(node_removed.parent_node)
            #compute the error with the resulting list of terminal nodes
            error_i = 0.0
            for o in nodes_i:
               error_i += (o.compute_error(o.Y_train,True) + alpha)
            
            #if we haven't get computed an error for this round of ranks
            #then make this the best option for the rank
            if not found_better_for_rank:
               best_prune_error_for_rank = error_i
               best_terminal_node_list_for_rank = copy(nodes_i)
               #next time we will have to compare to this error
               found_better_for_rank = True 
            elif error_i < best_prune_error_for_rank:
               #we found something better in this round of ranks
               best_prune_error_for_rank = error_i
               best_terminal_node_list_for_rank = copy(nodes_i)
         #we have exhauseted the current possible 1-node replacements with parents
         #see if this rank afforded a better error than the last
         if found_better_for_rank:  #we have something valid to compare to the previous rank's result
            if best_prune_error_for_rank < best_prune_error:
               #we have a set of nodes that decreases the error
               best_terminal_node_list = copy(best_terminal_node_list_for_rank)
               best_prune_error = best_prune_error_for_rank
               #we will continue to try smaller rank trees that a subsets of this one
            else:
               break #this rank offered no improvement in error. we know how to prune
         else:
            break #this rank offered no valid options probably because we were already at a single node last rank.
      
      #we have our winner or this next part does nothing
      self.terminal_nodes = copy(best_terminal_node_list)
      #make these nodes terminal, removing their branches if they have them
      for node in self.terminal_nodes:
         if not node.is_terminal:
            node.is_terminal = True
            node.cut_index = -1
            node.cut_value = 0.0
            node.branches = []
            node.determined_opt_cut = False
      #recompute the number of observations in each node
      self.n_obs_term_node = []
      for i in self.terminal_nodes:
         self.n_obs_term_node.append(i.obs_for_node())
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
   
   def return_deepcopy(self,the_parent):
   #{
      #this function is somewhat unfortunate because we cant use standard __deepcopy__()
      #this is because __deepcopy__() doesn't allow us to pass the reference to the parent node
      #(at least without screwing with memo, which I am hesitant to do)
      #we make copies of X and Y in constructor so just pass references in below
      the_copy = tree_node(copy.deepcopy(self.value),self.X_train,self.Y_train,copy.deepcopy(self.is_classification),the_parent) 
      
      #this is set in init but just to be explicit. we are not going to copy over other opt_cut members
      #they can be recomputed if needed
      the_copy.determined_opt_cut = False
      
      #if we are not a terminal node, we need to copy our actual cut information and copy the branches
      if not self.is_terminal:
         the_copy.is_terminal = False
         the_copy.cut_index = self.cut_index
         the_copy.cut_index = self.cut_index
         the_copy.branches = []
         the_copy.branches.append(self.branches[0].return_deepcopy(the_copy))
         the_copy.branches.append(self.branches[1].return_deepcopy(the_copy))

      return the_copy
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
   
   def return_list_terminal_nodes(self):
   #{
      #return a list of all terminal nodes that are descendants of this node
      if self.is_terminal:
         return [self]
      else:
         return [self.branches[0].return_list_terminal_nodes()] + [self.branches[1].return_list_terminal_nodes()]
   #}
   
   def is_progeny(self,other_node):
   #{
      #return True if other_node is 
      #a branch of this node, or the branch of a 
      #branch of this node etc.
      
      if self.is_terminal:
         #we have no progeny to check against
         return False
      else:
         #check our branches
         if (self.branches[0] is other_node) or (self.branches[1] is other_node):
            #it is one of our branches
            return True
         #check our branches branches
         else:
            return (self.branches[0].is_progeny(other_node) or self.branches[1].is_progeny(other_node))
   #}

#}

