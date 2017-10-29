import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt

trset=np.loadtxt('input.txt',ndmin=2)
trset=trset.reshape(30,3)
norms=lin.norm(trset,ord=2,axis=1)
norms=norms.reshape(30,1)
trset=trset/norms

class Parzen:
 def __init__(self, category, data):
  self.category = category
  self.data = data

 def AddTrainData(self,data):
  self.data = np.concatenate((self.data,data))

 def CalcOutput(self,input,sigma,num):
  arg = (np.sum(self.data[:]*input,1) - 1)/sigma
  return np.sum(np.exp(arg))/(num * np.sqrt(2*np.pi) * sigma * sigma) 

Set1 = Parzen(1,trset[0:10][:])
Set2 = Parzen(2,trset[10:20][:])
Set3 = Parzen(3, trset[20:30][:])
Sets = [Set1,Set2,Set3]

def Solve(sigma, num, inputs, sets):
 res=[]
 norms=lin.norm(inputs,ord=2,axis=1)
 norms=norms.reshape(len(norms),1)
 inputs=inputs/norms
 for x in inputs:
  output=[]
  for lt in sets:
   output.append(lt.CalcOutput(x,sigma,num))
  res.append((1 + np.argmax(output), max(output)))
  output.clear()
 return res