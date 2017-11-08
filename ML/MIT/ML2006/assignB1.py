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
 theta = X[0,:] * y[0]
 for i in range(1,len(y)):
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
 
def geom_margin(X,theta):
 norm = np.sqrt(np.sum(theta**2))
 theta = theta.reshape(1,2)
 return np.min(np.abs(np.sum(X * theta,1)/norm))

def plot_perc(X,theta):
 t = theta.reshape(1,2).copy()
 sign = np.sign(np.sum(X*t,1))
 lx = sign >=0
 X1 = X[lx==True,:]
 X2 = X[lx==False,:]
 X3 = np.array([[0,0],theta])
 plt.plot(X1[:,0],X1[:,1],'ro',X2[:,0], X2[:,1], 'bo')#,X3[:,0],X3[:,1])
 plt.show()
 
 

Thetaa0 = np.array([1.0,1.0])
(Thetaa, Ka) = perceptron_train(trsetAX, trsetAY,Thetaa0)
Erra = perceptron_test(trsetAX, trsetAY, Thetaa)
one_a = Thetaa / np.sqrt(np.sum(Thetaa**2))
cos_a = one_a[0]

Thetab0 = np.array([1.0,1.0])
(Thetab, Kb) = perceptron_train(trsetBX, trsetBY,Thetab0)
Errb = perceptron_test(trsetBX, trsetBY, Thetab)
one_b = Thetab / np.sqrt(np.sum(Thetab**2))
cos_b = one_b[0]

Info = [['K=  ', Ka, Kb], ['Err=  ', Erra, Errb], ['Theta=  ', Thetaa, Thetab], ['Cos=  ', cos_a, cos_b]]
Ga = geom_margin(trsetAX, Thetaa)
Gb = geom_margin(trsetBX, Thetab)
Info.append(['G=  ', Ga, Gb])
Info.append(['R=  ', np.max(np.sum(trsetAX**2,1)), np.max(np.sum(trsetBX**2,1))])

Info =str(Info)
print(Info)

plot_perc(trsetAX,Thetaa)
plot_perc(trsetBX,Thetab)


