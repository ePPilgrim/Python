import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import misc
import scipy.ndimage.interpolation as it

def shift(img, pxl):
 f = it.shift(img,(0,-pxl),cval=1.0)
 return f

def dist(img1,img2):
 return np.sqrt(np.sum((img1-img2)**2))

def plotDist(img):
 d = np.zeros(9)
 for i in range(1,10):
  img2 = shift(img,i)
  d[i-1]=(dist(img,img2))
 plt.plot(np.arange(9)+1,d)
 plt.show()
 
def GetTangentMat(img):
 tv = np.ravel(img - img)
 for i in range(1,3):
  f = np.ravel(shift(img,i)-img)
  tv = np.vstack((tv,f))
 return tv

def PlotTangeDist(img):
 tv = GetTangentMat(img)
 tv1 = tv[2,:] + np.ravel(img)
 d = np.zeros(9)
 for i in range(1,10):
  tv2 = np.ravel(shift(img,i))
  d[i-1] = np.sqrt(np.sum((tv1 - tv2)**2))
 plt.plot(np.arange(9)+1,d)
 plt.show()

img = misc.imread('./for.png',flatten=True,mode='F')
img = img/np.max(img)

PlotTangeDist(img)

