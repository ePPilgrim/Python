
import numpy as np


def Alilang(n, m):
    prime = 100000007
    r = 1 + int(np.log2(n))
    c = n - n//2
    mat = np.array([[1] * c] + [list(range(1 + n//2,n + 1))] * r, 
                   dtype = 'uint64')
    mat[1:,:] = mat[1:,:] // np.cumprod([[2]] * r, 0) - 1
    mat = np.cumprod(mat, 
                     axis = 0, 
                     dtype = 'uint64' ) % prime
    fac = np.sum(mat, 
                 axis = 1, 
                 dtype = 'uint64') % prime
#    for i in range(1, mat.shape[0]):
#        mat[i,:] = (mat[i,:] * mat[i - 1,:]) % prime
#    fac = np.array([0] + [0] * r, 
#                   dtype = 'uint64')
#    for i in range(mat.shape[1]):
#        fac = (fac + mat[:,i]) % prime
    res = np.array([1] + [0] * r,
                   dtype = 'uint64')
    for i in range(m):
        s_res = np.roll(res, -1)
        val = np.sum(((res + s_res) * fac) % prime) % prime
        res = np.roll(res, 1)
        res[0] = val
    return res[0]


#v = [[5, 10], [4, 8], [4,11], [3,15], [4,15]]
v = [[1000, 500], [1234, 234], [2345, 345], [3456, 456], [4567, 567]]
for n, m in v:
    print(Alilang(n,m))


#6109294
#21604313
#40872650
#65497150
#24864254

#1550477
#5842
#170625
#2781184
#15346786

#T=1#int(input())
#for i in range(0, T):
# (N, M)=(5, 10)#(int(i) for i in input().split()) 
# res = Alilang(N,M) 
# print(res)

#1550477
#5842
#170625
#2781184
#15346786
   
  