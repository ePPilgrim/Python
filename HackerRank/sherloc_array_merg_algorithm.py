
def arrayMerging(m):
    prime = 1000000007
    m += [-1]
    N = len(m)
    arr2D = [[0]*N for i in range(N)]
    arr2D[N - 1][:] = [1] * len(arr2D[N - 1][:])
    fac = [i for i in range(N)]
    fac[0] = 1
    for i in range(1, N):
        fac[i] = (fac[i] * fac[i - 1]) % prime
    for i in range(N - 2,-1,-1):
            arr2D[i][1] = 1
            n = 1
            for j in range(i + 1,N):
                if m[j] < m[j - 1]: break
                n = j - i
                arr2D[i][n] = (n * arr2D[j][n]) % prime
            for j in range(n + 1, N):
                arr2D[i][j] = arr2D[i][n]
    print(arr2D)
    return sum(arr2D[0])
                
m_count = int(input())
m = list(map(int, input().rstrip().split()))
print(arrayMerging(m))