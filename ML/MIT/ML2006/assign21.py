import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt

class LinReg:
 def __init__(self,n):
  self.X = npr.uniform(-1.0, 1.0, (n, 3))
  self.y_true = (np.sum(np.tile([-10.0, -15.0, -7.5],(n,1)) * np.log(self.X),axis = 1) + 1)*2.0
  self.y_noise = self.y_true + 10.0*npr.randn(n)

