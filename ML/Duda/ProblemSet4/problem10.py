import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt

trset=np.loadtxt('input.txt',ndmin=2)
trset=trset.reshape(30,3)
norms=lin.norm(trset,ord=2,axis=1)
norms=norms.reshape(30,1)
trset=trset/norms

class RCE:
 def __init__(self, data,lam):
  self.data = data.reshape(3,10,3)
  self.dist = np.zeros(30)
  self.dist[:] = lam

 def TrainRCE(self):
  data = self.data
  dist = self.dist
  for i in range(0,data.shape[0]):
   lx = i != np.arange(3)
   for j in range(0,data.shape[1]):
    pt = data[i,j,:].reshape(1,1,3)
    dd = np.sum((data[lx,:,:] - pt)**2,2)
    dd = dd.reshape(dd.shape[0]*dd.shape[1])
    ind1 = np.argmin(dd)
    ind2 = np.ravel(np.arange(30).reshape(3,10)[lx,:])[ind1]
    dist[ind2] = min(dist[ind2],np.sqrt(dd[ind1]) - np.finfo(float).eps)
  self.dist = dist

 def ClassifyPoint(self,X):
  data = self.data
  for x in X:
   x = x.reshape(1,1,3)
   dist = np.sqrt(np.ravel(np.sum((data - x)**2,2)))
   lx = dist <= self.dist
   self.lx = lx
   lx = np.sum(lx.reshape(3,10),1)
   lx = (lx!=0)
   res = sum(lx)
   if res == 1: print(np.arange(3)[lx]+1)
   else: print ("Ambigous point")
   print(self.lx)

rce = RCE(trset,0.5)
rce.TrainRCE()
X = np.array([[0.53,-0.44,1.1],[-0.49,0.44,1.11], [0.51,-0.21,2.15]])
norms = lin.norm(X,ord=2,axis=1)
norms = norms.reshape(3,1)
X = X / norms

rce.ClassifyPoint(X)  
