import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.gaussian_process.kernels import RBF

def sampling(n,k,sigma):
 N = n + 20000
 d = sigma.size
 mean = np.zeros(d)
 cov = np.diag(sigma)
 X1 = npr.multivariate_normal(mean,cov**2,N)
 Dist1 = np.sum((X1 / sigma.reshape(1,d))**2,1)
 idx = np.argsort(Dist1)
 inidx = idx[:k]
 outidx = idx[k:]
 npr.shuffle(outidx)
 outidx = outidx[:n-k]
 X = X1[np.append(inidx,outidx),]
 X = np.hstack((X,np.ones((X.shape[0],1))))
 X[:k,-1] = -1.0
 npr.shuffle(X)
 return X

def train_kernel_perceptron(X, y, kernel):
 alpha = np.zeros(X.shape[0],dtype=np.int32)
 K = kernel(X,X)
 n = y.size
 mist = True
 iter = 0
 while (mist == True) and (iter < 100000):
  mist = False
  for i in range(n):
   t = y[i]*np.dot(alpha,K[:,i])
   if t <= 0:
    iter = iter + 1
    alpha[i]+=y[i]
    mist = True
 print(iter)
 return alpha

def discriminant_function(alpha, X, kernel, X_test):
 w = (alpha != 0)
 X_=kernel(X[w,:],X_test)
 ans = np.ravel(np.dot(alpha[w],X_))
 return ans

class PK:
 def __init__(self,c=1.0, d=2):
  self.c = c
  self.d = d

 def __call__(self, X, Y):
  return (np.dot(X,Y.T) + self.c)**self.d

def solve(betta,dim):
 n = 4000
 k = 3000
 sigma = np.array([16.0,1.0])
 X = sampling(n,k,sigma)
 y_train = X[:,-1]
 X_train = X[:,:2]
 X = sampling(n,k,sigma)
 y_test = X[:,-1]
 X_test = X[:,:2]
 plt.plot(X_test[y_test <= 0,0], X_test[y_test<=0,1], 'ro', X_test[y_test > 0,0], X_test[y_test>0,1], 'go')
 plt.show()
 kern_poly = PK(1.0,dim)
 kern_rbf = RBF(betta)
 w_poly = train_kernel_perceptron(X_train,y_train, kern_poly)
 w_rbf = train_kernel_perceptron(X_train,y_train,kern_rbf)
 yy_poly = discriminant_function(w_poly, X_train, kern_poly,X_test)
 yy_rbf = discriminant_function(w_rbf, X_train, kern_rbf,X_test)
 plt.plot(X_test[yy_poly <= 0,0], X_test[yy_poly<=0,1], 'ro', X_test[yy_poly > 0,0], X_test[yy_poly>0,1], 'go')
 plt.show()
 plt.plot(X_test[yy_rbf <= 0,0], X_test[yy_rbf<=0,1], 'ro', X_test[yy_rbf > 0,0], X_test[yy_rbf>0,1], 'go')
 plt.show()

solve(1.0, 2)
