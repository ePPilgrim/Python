import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt

def linear_regress(y,X):
 return lin.multi_dot([lin.inv(np.dot(X.T,X)),X.T,y])

def linear_pred(thau, X):
 return np.dot(X,thau)

def feature_mapping(X, id):
 if id == 2:
  X = np.log(X*X)
 return np.hstack((X,np.ones(X.shape[0])[:,np.newaxis]))

def calc_mse(y,thau,X):
 return np.sum((np.dot(X, thau) - y)**2)/y.size

def active_learn(X,q,r):
 allidx = np.arange(X.shape[0])
 idx = np.arange(q)
 for i in range(r):
  idx_= np.setdiff1d(allidx,idx)
  A = X[idx,:]
  A = lin.inv(np.dot(A.T,A))
  V = np.dot(A,X[idx_,:].T)
  v1 = np.sum(V*V,axis=0)
  v2 = np.sum(1.0 + V * X[idx_,:].T, axis=0)
  v = v1/v2
  idx=np.append(idx,idx_[np.argmax(v)])
 return idx 

def randomly_select(X,q,r):
 idx = np.arange(q)
 idx_ = np.arange(q,X.shape[0])
 idx = np.append(idx,npr.choice(idx_,r))
 return idx 

class LinReg:
 def __init__(self,n=1000):
  self.X = npr.uniform(-1.0, 1.0, (n, 3))
  self.y_true = (np.sum(np.tile([-10.0, -15.0, -7.5],(n,1)) * np.log(self.X * self.X),axis = 1) + 2.0)
  self.y_noise = self.y_true + 10.0*npr.randn(n)

 def solvec(self):
  X1 = feature_mapping(self.X,1)
  X2 = feature_mapping(self.X,2)
  idx1 = active_learn(X1, 5, 10)
  idx2 = active_learn(X2, 5, 10)
  thau1 = np.ravel(linear_regress(self.y_noise[idx1],X2[idx1,:]))
  thau2 = np.ravel(linear_regress(self.y_noise[idx2],X2[idx2,:]))
  y1 = linear_pred(thau1, X2)
  y2 = linear_pred(thau2, X2)
  err1 = np.sum((self.y_true - y1)**2)/y1.size
  err2 = np.sum((self.y_true - y2)**2)/y2.size
  self.y1 = y1
  self.y2 = y2
  return (err1, err2)

 def solveb(self):
  X = feature_mapping(self.X,2)
  err = []
  for i in range(50):
   idx = randomly_select(X,5,10)
   thau = np.ravel(linear_regress(self.y_noise[idx],X[idx,:]))
   y = linear_pred(thau,X)
   err.append(np.sum((self.y_true - y)**2)/y.size)
  err1 = np.median(err)
  idx = active_learn(X, 5, 10)
  thau = np.ravel(linear_regress(self.y_noise[idx],X[idx,:]))
  y = linear_pred(thau, X)
  err2 = np.sum((self.y_true - y)**2)/y.size
  return (err1, err2)

obj = LinReg()
(err1, err2) = obj.solveb()

