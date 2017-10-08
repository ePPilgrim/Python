import sys

N=int(input())
K=int(input())
W=[]
for i in range(0,N):
 W.append(int(input()))

def solve(k,w):
 w.sort()
 w=[0]+w
 s=w[:]
 for i in range(1,len(s)):
  s[i]+=s[i-1]
 min=0
 for i in range(1,k):
  min+=s[k]-2*s[i]
 sum = min
 for i in range(k+1, len(s)):
  sum = sum - 2*(s[i-1] - s[i-k]) + (k-1)*(w[i]+w[i-k])
  if sum < min:
   min = sum
 return min

print(solve(K,W))


