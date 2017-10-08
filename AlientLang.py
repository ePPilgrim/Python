import sys

class Alilang:
 P = 100000000007
 def __init__(self,n,m):
  self.n = n
  self.m = m
  self.amount = []
  self.res = n - n//2
 
 def ddevide(self):
  n = self.n
  while n > 0:
   m = n//2
   self.amount.append([(m + 1)%2, n%2, n - m, n - m])
   n = m
  self.amount.append([0,0,0,0])
 
 def solve(self):
  self.ddevide()
  for i in range(1,self.m):
   for j in range(0,len(self.amount)-2):
    sum1 = self.amount[j+2][3]
    sum2 = self.amount[j+1][3] - sum1
    n = self.amount[j][2]
    if sum2<0:
     sum2 += self.P
    sum = (sum1*n)%self.P
    m = 1+n//2
    if tuple(self.amount[j][0:2]) == (1,0): 
     m = n//2
    self.amount[j][3] = ((sum1 + (m*sum2)%self.P)%self.P + (n*self.res)%self.P)%self.P
   self.amount[-2][-1]=self.res
   self.res = self.amount[0][-1]
  return self.amount[0][-1]

T=1#int(input())
for i in range(0, T):
 (N, M)=(5, 10)#(int(i) for i in input().split()) 
 lang = Alilang(N,M) 
 res = lang.solve()
 print(res)

 
   
  