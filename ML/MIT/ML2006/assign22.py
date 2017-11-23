import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.gaussian_process.kernels import RBF

boston = load_boston()

def get_train_test_random_samples(n_train, n_total):
 ind_total = np.arange(n_total)
 ind_train = np.unique(npr.randint(0,n_total,n_train))
 ind_test = np.unique(np.setdiff1d(n_total,ind_train))
 return (ind_train, ind_test) 

X = boston.data
y = boston.target

betta = 25.0

def solve(X, y, n_train, betta, n_lambdas):
 (ind_train, ind_test) = get_train_test_random_samples(n_train, X.shape[0]-1)
 X_train = np.copy(X[ind_train,:])
 y_train = np.copy(y[ind_train])
 X_test = np.copy(X[ind_test,:])
 y_test = np.copy(y[ind_test]) 
 kernel = RBF(betta)
 K = kernel(X_train)
 x = np.linspace(0.01,1.0, n_lambdas)
 one = np.eye(y_train.size)
 err_test = []
 err_train = []
 for lam in x:
  av = np.dot(lin.inv(lam * one + K),y_train).reshape(y_train.size,1)
  YY = kernel(X_train,X)
  yy = np.sum(YY*av,0)
  yy_test = yy[ind_test]
  yy_train = yy[ind_train]
  err_test.append(np.sum((yy_test-y_test)**2)/yy_test.size)
  err_train.append(np.sum((yy_train-y_train)**2)/yy_train.size)
 err_train = np.array(err_train)
 err_test = np.array(err_test)
 plt.plot(x, err_train,'ro',x,err_test,'go')
 plt.show()

