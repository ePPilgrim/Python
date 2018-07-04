
def candies(n, arr):
    W0 = max(arr) + 1
    arr=[W0] + arr + [W0]
    w = [[] for i in range(W0 + 2)]
    I0 = 1
    for i in range(1,n + 1):
        w[arr[i]].append(i)
        if I0 > arr[i]: I0 = arr[i]
    for i in range(I0, W0):
        if len(w[i]) != 0: w[i].append(n+1)     
    ans = [[1] for i in range(n)]
    I0 += 1
    for i in range(I0,len(w)):
        if len(w[i])!= 0:
            i1 = w[x][0]
            for j in range(1, len(w[i])):
                if w[i][j] == w[i][j-1] + 1: 
                    continue
                i2 = w[i][j] - 1
                
            
                
                
            