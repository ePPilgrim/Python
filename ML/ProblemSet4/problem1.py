import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

n = 20000
pts = npr.random((n, 3)) - 0.5
num_x = 1000
h_x = np.linspace(0.001, 1, num_x)
dest=np.array([np.count_nonzero(np.all(np.abs(pts)<=i,1)) for i in h_x])
denum = n * (h_x**3)
y = dest / denum
plt.plot(h_x,y)
plt.show()




