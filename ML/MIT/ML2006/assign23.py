import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.gaussian_process.kernels import RBF

def sampling(n,k,sigma):
 N = n + 20000
 mean = np.zeros(sigma.size)
 cov = np.diagonal(sigma)
 X1 = npr.multivariate_normal(mean,sigma**2,N)
 Dist1 = np.sum((X1 / sigma.reshape(1,sigma.size))**2,1)
 idx = np.argsort(Dist1)
 inidx = idx[:k]
 outidx = npr.shuffle(idx[k:])[:n-k]
 X = X1[np.append(inidx,outidx),]
 X = np.hstack((X,np.ones((X.shape[0],1))))
 X[:k,-1] = -1.0
 npr.shuffle(X)
 return X

def train_kernel_perceptron(X, y, kernel):
 alpha = np.zeros(X.shape[0],dtype=np.int32)
 K = kernel(X)
 n = y.size
 mist = True
 while mist:
  mist = False
  for i in range(n):
   if y[i]*np.dot(alpha,K[:,i]) <= 0:
    alpha[i]+=1
    mist = False
 return alpha
 
def discriminant_function(alpha, X, kernel, X_test):
 w = (alpha != 0)
 X_=kernel(X[w,:],X_test)
 ans = np.ravel(np.dot(w,X_))
 return ans
 
 
