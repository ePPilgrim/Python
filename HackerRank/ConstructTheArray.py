import sys

def countArray(n, k, x):
    prime = int(1000000007)
    dp0 = 1
    dp1 = 0 
    for i in range(n-2):
        temp = (dp1 + (((k - 2) * dp0) % prime)) % prime
        dp1 = ((k - 1) * dp0) % prime
        dp0 = temp
    if x == 1:
        return dp1
    else:
        return dp0
    

if __name__ == "__main__":
    n, k, x = input().strip().split(' ')
    n, k, x = [int(n), int(k), int(x)]
    answer = countArray(n, k, x)
    print(answer)

