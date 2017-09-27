from array import array

n=int(input())
m=int(input())
c=array('Q',[int(i) for i in input().split()])

def solve(n, c):
w=array('Q',[0 for i in range(0,n+1)])
for i in c:
k=n//i

