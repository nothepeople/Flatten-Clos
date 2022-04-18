import heapq
import copy


INF = 1024
"""
FIND ALL SHORTEST PATHS BETWEEN ANY SRC-DST PAIRS
"""
def dijkstra_algo(topo, switches, src):
  pq = []
  pre = [[] for _ in range(switches)]
  visited = [False for _ in range(switches)]
  dis = [INF for _ in range(switches)]
  dis[src] = 0
  ## pair: (distance, u) src->u 
  # print(dis)
  heapq.heappush(pq, [0, src])
  
  while len(pq) > 0:
    pair = heapq.heappop(pq)
    u = pair[1]
    # print("loops")
    if visited[u]: continue
    visited[u] = True
    
    for v in range(switches):

      if topo[u][v] != 1: continue
      # print("v-{}".format(v))
      if dis[v] > dis[u] + topo[u][v]:
        # print("aoo")
        dis[v] = dis[u] + topo[u][v]
        del pre[v][:]
        pre[v].append(u)
        if not visited[v]: heapq.heappush(pq, [dis[v], v])
      elif dis[v] == dis[u] + topo[u][v]:
        pre[v].append(u)
        if not visited[v]: heapq.heappush(pq, [dis[v], v])
  # print(pre)
  return pre

def dfs(pre, paths, path, src, dst, cur):
  if cur == src:
    path.append(src)
    # print("dfs {}".format(path))
    deepPath = copy.deepcopy(path)
    deepPath.reverse()
    paths.append(deepPath)

    path.pop()
    return
  
  for i in range(len(pre[cur])):
    path.append(cur)
    dfs(pre, paths, path, src, dst, pre[cur][i])
    path.pop()

def cal_shortest_path(topo, switches):
  shortest_paths_of_any_pair = {}
  for i in range(switches):
    shortest_paths_of_any_pair[i] = {}
    for j in range(switches):
      shortest_paths_of_any_pair[i][j] = []

  pairs = 0
  sum_path_length = 0
  sum_path_nums = 0
  for src in range(switches):
    pre = dijkstra_algo(topo, switches, src)
    for dst in range(switches):
      paths = []
      path = []
      dfs(pre, paths, path, src, dst, dst)
      # print(paths)
      shortest_paths_of_any_pair[src][dst] = paths
      pairs += 1
      sum_path_length += len(paths[0])
      sum_path_nums += len(paths)
  print("Number of ECMP paths: %.2f, Avg shortest path len: %.2f" % (sum_path_nums / pairs, sum_path_length / pairs))
  return shortest_paths_of_any_pair

"""
--SHORTEST PATH BETWEEN ANY SRC-DST PAIR--
    0
  / |  \ 
 1  2   3
 \  /\  /
  4    5
   \  /
    6
topo_matrix = [
    [0, 1, 1, 1, INF, INF, INF],
    [1, 0, INF, INF, 1, INF, INF],
    [1, INF, 0, INF, 1, 1, INF],
    [1, INF, INF, 0, INF, 1, INF],
    [INF, 1, 1, INF, 0, INF, 1],
    [INF, INF, 1, 1, INF, 0, 1],
    [INF, INF, INF, INF, 1, 1, 0]
]
"""
def test_cal_shortest_path():
  topo_matrix = [
    [0, 1, 1, 1, INF, INF, INF],
    [1, 0, INF, INF, 1, INF, INF],
    [1, INF, 0, INF, 1, 1, INF],
    [1, INF, INF, 0, INF, 1, INF],
    [INF, 1, 1, INF, 0, INF, 1],
    [INF, INF, 1, 1, INF, 0, 1],
    [INF, INF, INF, INF, 1, 1, 0]
  ]
  switches = 7
  sp_path = cal_shortest_path(topo_matrix, switches)
  print(sp_path[0][6])
  print(sp_path[5][0])

if __name__ == "__main__":
    test_cal_shortest_path()