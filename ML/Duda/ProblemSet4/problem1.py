import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt

def fun_a(n):
 pts = npr.random((n, 3)) - 0.5
 return pts

def fun_b(pts, nx,n):
 x = np.linspace(0.001, 1, nx)
 dest=np.array([np.count_nonzero(np.all(np.abs(pts)<=i,1)) for i in x])
 denum = n * (x**3)
 y = dest / denum
 plt.plot(x,y)
 plt.show()

def fun_c(pts,n):
 x = lin.norm(pts,ord=2,axis=1)
 x.sort()
 y = 3*(np.arange(n)+1)/(4*n*np.pi*(x**3))
 plt.plot(x,y)
 plt.show()

def fun_d(n):
 pts = npr.randn(n,3)
 return pts

n = 20000
nx = 1000






