
def candies(n, arr):
    arr=[n+1000] + arr + [n + 1000]
    w = [[] for i in range(100001)]
    for i in range(1,n + 1):
        w[arr[i]].append(i)
    ans = [[1] for i in range(n-2)]
    for x 