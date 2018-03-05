import sys


prime = int(10e9 + 7)


def kingdomDivision(roads):
    (res, sink) = Solve(1, roads)
    return (res + res) % prime


def Solve(graph):
    while True:
        x, y = 0, 0
        for i in graph:
            if len(graph[i][2]) == 1:
                x = graph[i][0]
                y = graph[i][1]
                x = (prime + y - x) % prime
                ik = graph[i][2][0]
                graph[ik][0] = (graph[ik][0] * x) % prime
                graph[ik][1] = (graph[ik][1] * ((x + y) % prime)) % prime
                graph[i][2] = []
                graph[ik][2].remove(i)
            elif 
              
    

if __name__ == "__main__":
    n = int(input().strip())
    roads = [[[1],[1],[]] for i in range(n + 1)]
    for roads_i in range(n-1):
       roads_t = [int(roads_temp) for roads_temp in input().strip().split(' ')]
       roads[roads_t[0]][2].append(roads_t[1])
       roads[roads_t[1]][2].append(roads_t[0])
    print(kingdomDivision(roads))
