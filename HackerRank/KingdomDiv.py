import sys


prime = 1000000007


def kingdomDivision(graph, n):
    leaves = []
    for i in range(n):
        if len(graph[i][2]) == 1:
           leaves.append(i)
    while len(leaves) != 1 :
        nextleaves = []
        for i in leaves:
            x = (prime + graph[i][1] - graph[i][0]) % prime
            y = graph[i][1]
            if len(graph[i][2]) == 0:
                nextleaves.append(i)
                continue
            j = graph[i][2].pop()
            graph[j][0] = (graph[j][0] * x) % prime
            graph[j][1] = (graph[j][1] * ((x + y) % prime)) % prime
            graph[j][2].remove(i)
            if len(graph[j][2]) == 1 :
                nextleaves.append(j)
        leaves = nextleaves
    ik = leaves[0]
    ans = (prime + graph[ik][1] - graph[ik][0]) % prime
    return (ans + ans) % prime
      
    
if __name__ == "__main__":
    n = int(input().strip())
    roads = [[1,1,set()] for i in range(n)]
    for roads_i in range(n-1):
       roads_t = [int(roads_temp) for roads_temp in input().strip().split(' ')]
       roads[roads_t[0] - 1][2].add(roads_t[1] - 1)
       roads[roads_t[1] - 1][2].add(roads_t[0] - 1)
    print(kingdomDivision(roads, n))
