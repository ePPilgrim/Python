import numpy as np


class TriFunc:
 def __init__(self, m, sigma,left=False,right=False):
  self.m = m
  self.sigma = sigma
  self.left = left
  self.right = right

 def __call__(self,x):
  y = 1
  if (self.left or self.right) is False:
   t = np.abs(x-self.m)/self.sigma
   if t < 1: y = 1 - t
   else: y = 0
  elif self.left is True:
   t = (x - self.m)/self.sigma
   if t > 0 and t < 1: y = 1 - t
   elif t >= 1: y = 0
  else:
   t = (self.m - x)/self.sigma
   if t > 0 and t < 1: y = 1 - t
   elif t >= 1: y = 0
  return y

#Cherry {small,spherical,red}
w1=[TriFunc(2,3,left=True), TriFunc(1.1,0.2,right=True), TriFunc(0.9,0.3,right=True)]
#Orange {medium,spherical, orange}
w2=[TriFunc(4,3), TriFunc(1.1,0.2,right=True), TriFunc(0.5,0.3)]
#Banana {large,thin,yellow}
w3=[TriFunc(6,3), TriFunc(2,0.6,left=True), TriFunc(0.1,0.1,left=True)]
#Peach {medium, spherical, orange-red}
w4=[TriFunc(4,3), TriFunc(1.1,0.2,right=True), TriFunc(0.7,0.3)]
#Plum {medium, spherical, red}
w5=[TriFunc(4,3), TriFunc(1.1,0.2,right=True), TriFunc(0.9,0.3,right=True)]
#Lemon {medium, oblong, yellow}
w6=[TriFunc(4,3), TriFunc(1.6,0.3), TriFunc(0.1,0.1,left=True)]
#Grapefruit {medium, spherical, yellow}
w7=[TriFunc(4,3), TriFunc(1.1,0.2,right=True), TriFunc(0.1,0.1,left=True)]

W = [w1, w2, w3, w4, w5, w6, w7]
Fruits = ['Cherry', 'Orange', 'Banana', 'Peach', 'Plum', 'Lemon', 'Grapefruit'] 

def Calc(x):
 res=[]
 global W
 for i in W:
  d = i[0](x[0])
  d *= i[1](x[1])
  d *= i[2](x[2])
  res.append(d)
 return np.array(res)
 
def ClassifyA(X):
 global Fruits
 for x in X:
  ind = np.argmax(Calc(x))
  print(Fruits[ind])


x1 = np.array([2.5,1.0,0.95])
x2 = np.array([7.5,1.9,0.2])
x3 = np.array([5.0,0.5,0.4]) 

X = [x1,x2,x3]

print("AAAAAAAAAAAAA")
ClassifyA(X)

L = np.zeros(49).reshape(7,7)
L[1:,0] = [1,1,0,2,2,1]
L[2:,1] = [2,2,0,0,1]
L[3:,2] = [1,0,0,2]
L[4:,3] = [2,2,2]
L[5:,4] = [1,1]
L[6:,5] = [2]
L = L + np.transpose(L)

def ClassifyB(X):
 global Fruits
 global L
 for x in X:
  f = Calc(x)
  f = f.reshape(1,f.size)
  ind = np.argmin(np.sum(L*f,1))
  print(Fruits[ind])

print("BBBBBBBBBBBBBB:")
ClassifyB(X)
  
 








