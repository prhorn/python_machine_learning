import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

#K-means culustering based on
#unweighted euclidean distance

class kmeans:
#{
   #  K     -- the number of clusters we are seeking
   #
   #  X     -- our training data. this will be reordered
   #           so that we can grab points assigned to each cluster
   #           by using row offsets based on N_k
   #
   #  N     -- total number of obs, rows of X
   #
   #  m     -- number of features/predictors, cols of X
   #
   #  N_k   -- the number of training obs in each cluster
   #
   #  mu_k  -- the mean vector for each cluster,  K x m matrix 
   #
   
   def __init__(self,X_,K_):
   #{
      self.tolerance = 1.0E-7
      self.K = K_
      self.X = np.array(X_)

      if len(self.X.shape) == 1:
         #make sure X is always considered 2-D
         self.X = np.reshape(self.X,(self.X.shape[0],1))
      
      self.N = self.X.shape[0]
      self.m = self.X.shape[1]
      
      #select the K initial mean vectors as K different training data points
      initial_mean_vec_indices = np.random.choice(self.N,self.K,replace=False)
      self.mu_k = np.zeros((self.K,self.m))
      for i in range(self.K):
         self.mu_k[i,:] = np.array(self.X[initial_mean_vec_indices[i],:])
      
      self.assign_to_clusters() #reorders X and populates N_k
   #}
   
   def assign_to_clusters(self):
   #{
      cluster_assignment = [0]*self.N
      for i in range(self.N):
         best_cluster_i = 0
         temp = self.X[i,:] - self.mu_k[0,:]
         best_sq_dist_i = np.dot(temp,temp)
         for k in range(1,self.K): #we already did 0
            temp = self.X[i,:] - self.mu_k[k,:]
            temp_dist = np.dot(temp,temp)
            if temp_dist < best_sq_dist_i:
               best_sq_dist_i = temp_dist
               best_cluster_i = k
         cluster_assignment[i] = best_cluster_i
      
      sort_indices = np.argsort(cluster_assignment)
      self.X = self.X[sort_indices,:]
      self.N_k = np.bincount(cluster_assignment)

   #}
   
   def update_mean_vectors(self):
   #{
      row_off = 0
      for k in range(self.K):
         if self.N_k[k] > 0:
            self.mu_k[k,:] = (1.0/float(self.N_k[k]))*np.sum(self.X[row_off:row_off+self.N_k[k],:],axis=0)
            row_off = row_off + self.N_k[k]
         else:
            print 'no observations assigned to cluster k= '+str(k)
   #}

   def next_step(self):
   #{
      #we will converge on mean vectors not changing
      mu_last = np.array(self.mu_k)
      
      self.assign_to_clusters()
      self.update_mean_vectors()

      err = np.linalg.norm(self.mu_k - mu_last,ord='fro')
      print err
      if err < self.tolerance:
         return True
      
      return False
   #}
   
   def train(self):
   #{
      max_iter = 1000
      cur_iter = 0
      converged = False
      for i in range(max_iter):
         cur_iter = cur_iter + 1
         print 'kmeans iteration ',cur_iter
         converged = self.next_step()
         if converged:
            break
      if converged:
         print 'converged in '+str(cur_iter)+' iterations'
         return True
      else:
         print 'did not converge in '+str(cur_iter)+' iterations'
         return False
   #}

   def value(self):
   #{
      #return objective function value with current mean vectors
      #make sure we have the corresponding cluster assignment correct
      #(we converged after computing new mean vectors and deciding
      #  that they did not change much.)
      self.assign_to_clusters()

      obj = 0.0
      row_off = 0
      for k in range(self.K):
         if self.N_k[k]>0:
            temp = self.X[row_off:row_off+self.N_k[k],:] - self.mu_k[k,:]
            obj_k = 0.0
            for i in range(self.N_k[k]):
               obj_k = obj_k + np.dot(temp[i,:],temp[i,:])
            obj = obj + obj_k*float(self.N_k[k])
            row_off = row_off + self.N_k[k]
      
      return obj
   #}

   def predict(self,X_predict,plot=False):
   #{
      if len(X_predict.shape)==1:
         X_pred = np.reshape(X_predict,(X_predict.size,1))
      else:
         X_pred = np.array(X_predict)

      cluster_assignment = np.zeros(X_pred.shape[0],dtype=np.int) 
      for i in range(X_pred.shape[0]):
         best_cluster_i = 0
         temp = X_pred[i,:] - self.mu_k[0,:]
         best_sq_dist_i = np.dot(temp,temp)
         for k in range(1,self.K): #we already did 0
            temp = X_pred[i,:] - self.mu_k[k,:]
            temp_dist = np.dot(temp,temp)
            if temp_dist < best_sq_dist_i:
               best_sq_dist_i = temp_dist
               best_cluster_i = k
         cluster_assignment[i] = best_cluster_i
      

      if plot and (self.m == 2):
         sort_indices = np.argsort(cluster_assignment)
         X_sorted = X_pred[sort_indices,:]
         N_sorted = np.bincount(cluster_assignment)
         colors = itertools.cycle(["r","g","b","c","m"])
         fig = plt.figure()
         ax = fig.add_subplot(111)
         row_off = 0
         for k in range(self.K):
            #plot the mean vector
            color_k = next(colors)
            ax.scatter(self.mu_k[k,0],self.mu_k[k,1],marker="*",color=color_k,label=str(k)+' mean')
            if N_sorted[k]>0:
               ax.scatter(X_sorted[row_off:row_off+N_sorted[k],0],X_sorted[row_off:row_off+N_sorted[k],1],color=color_k,label=str(k))
               row_off = row_off + N_sorted[k] 
         plt.legend(loc='best')
         plt.show()
      
      return cluster_assignment
   #}

#}
