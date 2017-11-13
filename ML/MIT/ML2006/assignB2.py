import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers

trsetAX=np.loadtxt('./data/p1_a_X.dat',ndmin=2)
trsetAY=np.loadtxt('./data/p1_a_y.dat',ndmin=1)
trsetBX=np.loadtxt('./data/p1_b_X.dat',ndmin=2)
trsetBY=np.loadtxt('./data/p1_b_y.dat',ndmin=1)

class SVM:
 def __init__(self,X,y):
  self.X = X.copy()
  self.y = y.copy()

 def solve(self):
  P = matrix(np.diag([1,1,0]), tc='d')
  q = matrix(np.zeros(3),tc='d')
  h = np.ones(self.X.shape[0]).reshape(self.X.shape[0],1)
  y = -1.0*self.y.reshape(self.y.size,1)
  G = matrix(y * np.hstack((self.X,h)))
  h = matrix(-1.0*h)
  sol = solvers.qp(P,q,G,h)
  theta = np.array(sol['x'])
  self.theta = theta[0:2,0]
  self.shift = theta[2,0]
  theta = self.theta.reshape(1,2)
  shift = self.shift
  res = np.sum(self.y * (np.sum(self.X * theta,1) + shift) < 0)
  norm = np.sqrt(np.sum(theta**2))
  theta = theta.reshape(1,2)
  minn = np.min(np.abs(np.sum(self.X * theta,1)/norm))
  return (res, minn)

objA = SVM(trsetAX, trsetAY)
res, min= objA.solve()
print(res)
print(min)
print('B sets---------------------')
objB = SVM(trsetBX, trsetBY)
resB, minB = objB.solve()
print(resB)
print(minB)