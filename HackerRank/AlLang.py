import sys
import math

def Alilang(n, m):
    prime = 100000007
    r = 1 + int(math.log2(n))
    c = n - n//2
    mat = [[1] * c] + [list(range(1 + n//2,n + 1))] * r
    
    mat[1:,:] = mat[1:,:] // np.cumprod([[2]] * r, 0) - 1
    mat = np.cumprod(mat, 
                     axis = 0, 
                     dtype = 'uint64' ) % prime
    fac = np.sum(mat, 
                 axis = 1, 
                 dtype = 'uint64') % prime
    res = np.array([1] + [0] * r,
                   dtype = 'uint64')
    for i in range(m):
        s_res = np.roll(res, -1)
        val = np.sum(((res + s_res) * fac) % prime) % prime
        res = np.roll(res, 1)
        res[0] = val
    return res[0]

T=int(input())
for i in range(0, T):
    N=int(input())
    M=int(input()) 
    res = Alilang(N,M) 
    print(res)