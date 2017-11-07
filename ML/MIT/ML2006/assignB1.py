import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt

trsetAX=np.loadtxt('./data/p1_a_X.dat',ndmin=2)
trsetAY=np.loadtxt('./data/p1_a_y.dat',ndmin=1)
trsetBX=np.loadtxt('./data/p1_b_X.dat',ndmin=2)
trsetBY=np.loadtxt('./data/p1_b_y.dat',ndmin=1)

def perceptron_train(X, y, theta):
 k = 0
 for i in range(0,len(y)):
  t = X[i,:] * y[i]
  if np.sum(t*theta) < 0:
   theta += t
   k +=1
 return (theta,k)

def perceptron_test(X, y, theta):
 k = 0
 theta = theta.reshape(1,2)
 res = np.sum(y * np.sum(X * theta,1) < 0)
 return res

Thetaa0 = np.array([1.0,1.0])
(Thetaa, Ka) = perceptron_train(trsetAX, trsetAY,Thetaa0)
Erra = perceptron_test(trsetAX, trsetAY, Thetaa)
one_a = Thetaa / np.sqrt(np.sum(Thetaa0**2))
cos_a = one_a[0]

Thetab0 = np.array([1.0,1.0])
(Thetab, Kb) = perceptron_train(trsetBX, trsetBY,Thetab0)
Erra = perceptron_test(trsetBX, trsetBY, Thetab)
one_b = Thetab / np.sqrt(np.sum(Thetab0**2))
cos_b = one_b[0]