
def candies(n, arr):
    W0 = max(arr) + 1
    arr=[W0] + arr + [W0]
    w = [[] for i in range(W0 + 2)]
    I0 = 1
    for i in range(1,n + 1):
        w[arr[i]].append(i)
        if I0 > arr[i]: I0 = arr[i]    
    ans = [1]*(n + 2)
    I0 += 1
    for i in range(I0,len(w)):
            for j in w[i]:
                if arr[j-1] < arr[j] : 
                    ans[j] = 1 + ans[j-1]
                if arr[j+1] < arr[j]:
                    ans[j] = max(ans[j], ans[j+1] + 1)
    return sum(ans[1:n+1])
                
            
                
ans = candies(10,[2,4,2,6,1,7,8,9,2,1])              
print(ans)