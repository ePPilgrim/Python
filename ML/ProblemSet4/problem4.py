import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Voronoi, voronoi_plot_2d

trset=np.loadtxt('input.txt',ndmin=2)
trset=trset.reshape(30,3)
norms=lin.norm(trset,ord=2,axis=1)
norms=norms.reshape(30,1)
trset=trset/norms


class VorClassifier:
 def __init__(self,vertexes, cat):
  rix = np.arrange(30)
  self.vor2d = Voronoi(trset[(rix<10)^(rix>=20),:2])
  self.cat = cat
  
 def getDenseVor(self):
  ridge_points = self.vor2d.ridge_points
  lx = 
  for i in np.max(ridge_points]
  



