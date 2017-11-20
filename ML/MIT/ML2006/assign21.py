import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt

def linear_regress(y,X):
 return lin.multi_dot([lin.inv(np.dot(X.T,X)),X.T,y])

def linear_pred(thau, X):
 return np.dot(X,thau)

def feature_mapping(X, case):
 if case == 2:
  X = np.log(X)*2.0
 return np.hstack((X,np.ones(X.shape[0])[:,np.newaxis])

def active_learn(X, k1, k2):
 allidx = np.arange(X.shape[0])
 idx = np.arange(k1)
 idx_= np.setdiff1d(allidx,idx)
 for i in range(k2):
  A = X[idx,:]
  A = lin.inv(np.dot(A.T,A))
  V = np.dot(A,X[idx_,:].T)
  v1 = np.sum(V*V,axis=0)
  v2 = np.sum(1.0 + V * X[idx_,:].T, axis=0)
  v = v1/v2
  
  
 

class LinReg:
 def __init__(self,n):
  self.X = npr.uniform(-1.0, 1.0, (n, 3))
  self.y_true = (np.sum(np.tile([-10.0, -15.0, -7.5],(n,1)) * np.log(self.X),axis = 1) + 1)*2.0
  self.y_noise = self.y_true + 10.0*npr.randn(n)

