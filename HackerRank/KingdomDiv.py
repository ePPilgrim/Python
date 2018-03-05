import sys


prime = int(10e9 + 7)

def kingdomDivision(roads):
    (res, sink) = Solve(1, roads)
    return (res + res) % prime


def Solve(nik, graph):
    node = graph[nik]
    graph[nik] = []
    war, peace, cnt = 1, 1, 0
    for ik in node:
        if graph[ik] == []: 
            continue
        cnt += 1
        x, y = Solve(ik, graph)
        peace = (peace * (x + y)) % prime
        war = (war * x) % prime
    if cnt != 0:
        return ((prime + peace - war) % prime, peace)
    return (0, 1)


def FindSolution():
    n = int(input().strip())
    roads = [[]] * (n + 1)
    for roads_i in range(n-1):
       roads_t = [int(roads_temp) for roads_temp in input().strip().split(' ')]
       roads[roads_t[0]].append(roads_t[1])
       roads[roads_t[1]].append(roads_t[0])
    roads=[[], [2,3],[1],[1,4,5],[3], [3]]
    return kingdomDivision(roads)
    

if __name__ == "__main__":
    print(FindSolution())
