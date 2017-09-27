from array import array

n=int(input())
m=int(input())
c=array('Q',[int(i) for i in input().split()])

def solve(n, c):
 w=array('Q',[0 for i in range(0,n+1)])
 for i in c:
  for j in range(n,i-1):
   w[j]+=sum(w[k] for k in range(j,i-1,i))
 return w[n]

 res=solve(n,c)
 
  

