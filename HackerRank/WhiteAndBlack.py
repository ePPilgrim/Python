# -*- coding: utf-8 -*-
import sys


def cnt(x, n):
    llen = 0
    for k in range(n):
        llen += 1 & (x >> k)
    return llen


def combGen(x,n,prob):
    for k in range(n):
        if x & 1 << k == 0 and x & 1 << n - k - 1 != 0:
            continue
        p = 1.0 / n 
        if x & 1 << k != 0 and x & 1 << n - k - 1 == 0:
            p = 2.0 / n
        mask = (1 << k) - 1
        yield ((mask & x) | ((x >> 1) & ~mask), p * prob)


def solve(s,n,k):
    x = 0
    for i in range(len(s)):
        if s[i] == 'W':
            x = x | 1 << i  
    comb = {x: 1.0}
    for i in range(k):
        temp = comb
        comb = {}
        for (key,val) in temp.items():
            subset = [ y for y in combGen(key, n - i, val)]
            for kk, p in subset:
                comb[kk] = comb.get(kk,0) + p
    llen = cnt(x, n)
    stat = [0.0] * (llen + 1)
    m = n - k
    for (key, p) in comb.items():
        stat[llen - cnt(key, m)] += p
    ans = 0.0
    for i in range(len(stat)):
        ans += i * stat[i]
    return ans


n,k = input().strip().split(' ')
n,k = [int(n),int(k)]
balls = input().strip()

print('{0:.10f}'.format(solve(balls,n,k)))

        
    