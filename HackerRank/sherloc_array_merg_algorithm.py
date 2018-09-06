
def arrayMerging(m):
    prime = 1000000007
    N = len(m)
    arr2D = [[0] * (N - i) for i in range(N)]
    pps = [N - 1]
    
    for i in range(N - 2, -1,-1):
        if m[i] > m[i + 1]: pps = [i] + pps
        else : pps = [pps[0]] + pps
    print(pps)    
    perm = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(i + 1):
            if j != 0: perm[i][j] = (perm[i - 1][j - 1] * (i + 1)) % prime
            else : perm[i][0] = i + 1
    print(perm)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, pps[i] + 2):
            k1 = j - i - 1
            if j < N : 
                k2 = pps[j] - j 
                if k2 > k1 : k2 = k1
                for k in range(k2 + 1):
                    arr2D[i][k1] = (arr2D[i][k1] + perm[k1][k] * arr2D[j][k]) % prime
            else: arr2D[i][k1] = 1
    print(arr2D)        
    ans = 0
    for val in arr2D[0]:
        ans = (ans + val) % prime
        
    return ans
                
m_count = int(input())
m = list(map(int, input().rstrip().split()))
print(arrayMerging(m))