
import sys
import numpy as np


def Alilang(n, m):
    prime = 100000000007
    pos = np.arange(n//2 + 1, n + 1,dtype = int)
    k = int(np.ceil(np.log2(n)))
    mat = np.zeros((k,len(pos)),dtype=int)
    #mat = mat + 
    
    

T=1#int(input())
for i in range(0, T):
 (N, M)=(5, 10)#(int(i) for i in input().split()) 
 lang = Alilang(N,M) 
 res = lang.solve()
 print(res)

 
   
  