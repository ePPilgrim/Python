import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Voronoi, voronoi_plot_2d

class SinglePoint: 
 def __init__(self, num):
  self.num = num
  self.trainset = npr.uniform(-0.5,0.5, (4,num,6))
  self.testset = npr.uniform(-0.5,0.5,(100,6))

 def calc(self):
  veccat = np.zeros((100,6))
  for i in range(0,100):
   point = self.testset[i,:].reshape(1,1,6)
   dist = self.trainset * point
   s = np.zeros(dist.shape[0:2])
   for j in range(0,6):
    s += dist[:,:,j]
    veccat[i,j] = np.argmin(np.min(s,axis=1))+1
  self.veccat = veccat

 def findCategory(self,point,r):
  point.reshape(1,1,6)
  r = r + 1
  dists = self.trainset[:,:,:r] * point[:,:,:r]
  dists = np.sum(dists, axis=2)
  return np.argmin(np.min(dists,1)) + 1

obj = SinglePoint(1000000)

