import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

trset=np.loadtxt('input.txt',ndmin=2)
trset=trset.reshape(30,3)
norms=lin.norm(trset,ord=2,axis=1)
norms=norms.reshape(30,1)
trset=trset/norms

class Knearest:
 def __init__(self,datam,datanum):
  self.datam=datam
  self.datanum=datanum
  self.minv = datam.min(axis=0)
  self.maxv = datam.max(axis=0)

 def GetLinSpace1D(self):
  x = np.linspace(self.minv, self.maxv,1000)
  y = x.reshape(x.size,1)
  datav = self.datam.reshape(1,self.datanum)
  y = np.abs(y - datav)
  y.sort()
  return (y,x)

 def Solve1D(self,kv):
  y,x = self.GetLinSpace1D()
  for  k in kv:
   Y =k/2*self.datanum*y[:,k-1]
   plt.plot(x,Y)
   plt.show()

 def GetLinSpace2D(self):
  x = np.linspace(self.minv[0], self.maxv[0],100)
  y = np.linspace(self.minv[1],self.maxv[1],100)
  X,Y = np.meshgrid(x,y)
  datav = self.datam.reshape(1,self.datanum,2)
  x = (x.reshape(x.size,1) - datav[:,:,0])**2
  y = (y.reshape(y.size,1) - datav[:,:,1])**2
  Z = 2.0*np.pi*(x.reshape(x.shape[0],1,x.shape[1]) + y.reshape(1,y.shape[0],y.shape[1]))*self.datanum
  Z.sort()
  self.X = X
  self.Y = Y
  self.Z = Z

 def Solve2D(self,k):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X = self.X
  Y = self.Y
  Z = k/self.Z[:,:,k-1]
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  ax.set_zlim(Z.min(), Z.max())
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()

 def Solve3D(self,kv,pts):
  z = np.sum((pts.reshape(pts.shape[0],1,3)-self.datam.reshape(1,self.datanum, 3))**2,axis=2)
  zz = z.copy()
  zz.sort()
  for k in kv:
   rv = zz[:,k]
   zzz = z <= rv.reshape(rv.size,1)
   ind=np.arange(10)
   zzz=np.concatenate((np.sum(zzz[:,ind],axis=1),np.sum(zzz[:,10+ind],axis=1),np.sum(zzz[:,20+ind],axis=1))).reshape(3,3)
   res = np.argmax(zzz,axis=0)+1
   print(res)

kv = np.array([1,3,5])
obj1 = Knearest(trset[20:30,0],10)
obj2 = Knearest(trset[10:20,0:2],10)
obj2.GetLinSpace2D()
obj3 = Knearest(trset,30)
pts=np.array([[-0.41,0.82,0.88],[0.14,0.72,4.1],[-0.81,0.61,-0.38]])

