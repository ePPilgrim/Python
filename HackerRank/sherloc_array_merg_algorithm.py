
def arrayMerging(m):
    prime = 1000000007
    N = len(m)
    arr2D = [[0] * (N - i) for i in range(N)]
    pps = [N - 1]
    for i in range(N - 2, -1):
        if m[i] > m[i + 1]: pps = [i] + pps
        else : pps = [pps[0]] + pps
    perm = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(i + 1):
            if j != 0: perm[i][j] = (perm[i - 1][j - 1] * (i + 1)) % prime
            else : perm[i][0] = i
    
    for i in range(N - 1, -1):
        for j in range(pps[i] - i + 1):
            
    return sum(arr2D[0])
                
m_count = int(input())
m = list(map(int, input().rstrip().split()))
print(arrayMerging(m))