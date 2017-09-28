from array import array

#n=int(input())
#m=int(input())
#c=array('Q',[int(i) for i in input().split()])
n=4
m=3
c=array('Q',[1,2,3])

def solve(n, c):
 w=array('Q',[0 for i in range(0,n+1)])
 #print(str(w))
 for i in c:
  #print(i)
  #print(n)
  for j in range(n,i-1,-1):
   #print(j)
   w[j]+=sum(w[k] for k in range(j,i-1,i))
   #print(w[j])
 return w[n]

res=solve(n,c)
print(str(res))
  