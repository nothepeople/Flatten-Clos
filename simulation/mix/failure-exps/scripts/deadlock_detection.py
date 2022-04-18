
import copy
import fc
import random
from ecmp import cal_shortest_path

"""
DEADLOCK DETECTION 
"""
class Port:
  def __init__(self, id, port_id, switch_id):
    self.id = id
    self.port_id = port_id
    self.switch_id = switch_id
    
  # def __hash__(self):
  #   return hash((self.port_id, self.switch_id))

  def __lt__(self, other):
    return self.switch_id < other.switch_id or (self.switch_id == other.switch_id and self.port_id == other.port_id)

  def __str__(self):
    return "id: {}, switch_id: {}, port_id: {}".format(self.id, self.switch_id, self.port_id)

def dfs_find_cycles(dependency_graph, cycles, visited, trace, node):
  if visited[node] == True:
    if node in trace:
      # trace_idx = trace.index(node)
      cycle = copy.deepcopy(trace)
      # time.sleep(0.1)
      # print(cycle)
      cycle.append(node)
      cycles.append(cycle)
      return 
    return
  visited[node] = True
  trace.append(node)

  for i in range(len(dependency_graph[node])):
    if dependency_graph[node][i] == 1:
      # print("nxt node:{}".format(i))
      dfs_find_cycles(dependency_graph, cycles, visited, trace, i)
  trace.pop()


def deadlock_detection(shortest_paths, switches, ports, ports_conn_matrix):
  ## PAUSE PROPAGATION 
  ## Switch PORT
  port_set = {}
  idx = 0
  for sid in range(switches):
    port_set[sid] = {}
    for pid in range(ports):
      port_set[sid][pid] = Port(id=idx, port_id = pid, switch_id = sid)
      idx += 1
  # print(idx)
  pause_propagations = []

  ## sp_idx means the shortest path index
  for sp_idx in range( len(shortest_paths) ):
    pause = []
    sp = shortest_paths[sp_idx]
    for i in range(len(sp) - 1, 0, -1):
      cur_sid = sp[i]
      last_sid = sp[i - 1]
      # print("last sid: {} cur sid{} {}".format(last_sid, cur_sid, ports_conn_matrix[last_sid][cur_sid]))
      # if len(ports_conn_matrix[last_sid][cur_sid]) == 0:
      #   continue
      input_port = ports_conn_matrix[last_sid][cur_sid][1]
      pause.append([cur_sid, input_port, sp_idx])
    pause_propagations.append(pause)

  ## construct dependency graph
  dependency_graph = [[0 for _ in range(idx)] for _ in range(idx)]
  corresponding_path_set = {}
  for i in range(idx):
    corresponding_path_set[i] = {}
    for j in range(idx):
      corresponding_path_set[i][j] = []

  for pause in pause_propagations:
    for i in range(len(pause) - 1):
      endpoint1 = pause[i]
      endpoint2 = pause[i+1]
      id1 = port_set[endpoint1[0]][endpoint1[1]].id
      id2 = port_set[endpoint2[0]][endpoint2[1]].id
      # print(id1, id2)
      dependency_graph[id1][id2] = 1
      corresponding_path_set[id1][id2].extend( [endpoint1[2], endpoint2[2] ])
  
  # print(dependency_graph)
  has_cycle = False
  results = []
  ## Detect cycle in dependency graph
  for pnode in range(idx):
    cycles = []
    visited = [False for _ in range(idx)]
    trace = []
    dfs_find_cycles(dependency_graph, cycles, visited, trace, pnode)
    # print(cycles)
    has_cycle = len(cycles) > 0
    if has_cycle:
      for cycle in cycles: results.append(cycle)
      break
  return has_cycle, results, port_set, corresponding_path_set

"""
How to map one cycle to corresponding routing paths ?
"""
def map_cycle_to_paths(corresponding_path_set, port_set, shortest_paths, ports_conn_matrix, switches, ports, one_cycle):
  possible_paths = []
  
  print("one cycle", one_cycle)
  for i in range( len(one_cycle) - 1):
    endpoint1 = one_cycle[i]
    endpoint2 = one_cycle[i + 1]
    for path_idx in corresponding_path_set[endpoint1][endpoint2]:
      possible_paths.append(shortest_paths[path_idx])

  print("before deduplicate pp paths", len(possible_paths))
  # possible_paths = set(possible_paths )
  deduplicate_set = []
  deduplicate_set = set(deduplicate_set)
  for pp in possible_paths:
    deduplicate_set.add(str(pp))
  # print("possible paths", deduplicate_set)

  possible_paths = []
  for unique_path in deduplicate_set:
    sids = unique_path.replace('[', '').replace(']', '').split(',')
    ll = []
    # print(sids)
    for sid in sids:
      # print(sid)
      ll.append(int(sid))
    possible_paths.append(ll)
  print("pp paths", len(possible_paths))
  

  print(possible_paths[0 : 12])

  eps = []
  for pp in possible_paths[0 : 12]:
    eps.extend( pp )
  # eps = set(eps)
  # print(eps)
  # print(eps = [])

  # for i in range(len(possible_paths) ):
  #   for j in range(i + 1, len(possible_paths)):
  #     ret, _, _, _ = deadlock_detection(possible_paths[i : j], switches, ports, ports_conn_matrix)
  #     if ret:
  #       print("{} - {} = {}".format(j, i, j - i))
  
  # for idx in sub:
  #   path_set.append(possible_paths[idx])


""" 
-------- TEST TOPOLOGY----------
      (0) 2 (1)                 
      /       \                 
     /         \                 
    /           \                
   (0)          (0)              
    0            1               
   (1)          (1)             
    \           /            
     \         /            
      \       /               
      (0) 3 (1)               
-------------------------------
"""

def test_deadlock_detection1():
  shortest_paths = [
    [0, 2, 1],
    [1, 3, 0],
    [3, 0, 2],
    [2, 1, 3]
  ]

  ports_conn_matrix={}
  for i in range(4):
    ports_conn_matrix[i] = {}
    for j in range(4):
      ports_conn_matrix[i][j] = []
  ports_conn_matrix[0][2] = [0, 0]
  ports_conn_matrix[0][2] = [0, 0]

  ports_conn_matrix[2][1] = [1, 0]
  ports_conn_matrix[1][2] = [0, 1]
  
  ports_conn_matrix[1][3] = [1, 1]
  ports_conn_matrix[3][1] = [1, 1]

  ports_conn_matrix[3][0] = [0, 1]
  ports_conn_matrix[0][3] = [1, 0]

  switches = 4
  ports = 2
  ret, cycles, port_set = deadlock_detection(shortest_paths, switches, ports, ports_conn_matrix)
  if ret:
    print("Have the risk of PFC-deadlock!!!")
  else:
    print("This routing has no risk of deadlock!!!")
  print("cycles", cycles)
  map_cycle_to_paths(port_set, shortest_paths, ports_conn_matrix, switches, ports, cycles[0])


"""
        (0) 0 (1)
        /       \ 
       /         \ 
      /           \ 
  (0) 1 (1)----(0) 2 (1)
"""
def test_deadlock_detection2():
  shortest_paths = [
    [0, 2, 1],
    [2, 1, 0],
    # [1, 0, 2]
  ]
  ports_conn_matrix={}
  for i in range(3):
    ports_conn_matrix[i] = {}
    for j in range(3):
      ports_conn_matrix[i][j] = []
  
  ports_conn_matrix[0][1] = [0, 0]
  ports_conn_matrix[1][0] = [0, 0]
  
  ports_conn_matrix[0][2] = [1, 1]
  ports_conn_matrix[2][0] = [1, 1]

  ports_conn_matrix[1][2] = [1, 0]
  ports_conn_matrix[2][1] = [0, 1]

  switches = 3
  ports = 2
  has_cycle = deadlock_detection(shortest_paths, switches, ports, ports_conn_matrix)
  if has_cycle:
    print("Have the risk of PFC-deadlock!!!")
  else:
    print("No risk of PFC-deadlock!!!")

def test_fc_routing():
  ports_of_vir_layer=[2,4,4,2]
  switches=64
  ports=16
  to_hosts=4
  layers=4
  topo_matrix, ports_conn_matrix, switch_objs = fc.topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  
  vclos = fc.convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  
  virtual_up_down_paths= fc.paths_of_any_pairs(vclos, layers, switches)

  path_set = []
  for i in range(switches):
    for j in range(i+1, switches):
      if len(virtual_up_down_paths[i][j]) == 0:
        print("error")
      for p in virtual_up_down_paths[i][j]:
        # print(p)
        path_set.append(p)
  ret = deadlock_detection(path_set, switches, ports, ports_conn_matrix)
  if ret:
    print("Deadlock Risk")
  else:
    print("No PFC-deadlock Risk!!!")

def test_fc_ecmp_deadlock():
  random.seed(1)
  switches=64
  ports=16
  to_hosts=4
  ports_of_vir_layer=[2,4,4,2]
  topo_matrix, ports_conn_matrix, _ = fc.topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  shortest_paths = cal_shortest_path(topo_matrix, switches)
  paths = []
  for src in range(switches):
    for dst in range(switches):
      if src == dst: continue
      for path in shortest_paths[src][dst]:
        paths.append(path)
  print("paths: {}".format( len(paths) ) )
  # print(shortest_paths[0])
  ret, cycles, port_set, corresponding_path_set = deadlock_detection(paths, switches, ports, ports_conn_matrix)
  # print(port_set)

  minLen = 0xfffff
  minIDx = 0xfffff
  for i in range(len(cycles)):
    if len(cycles[i]) < minLen:
      minLen = len(cycles[i])
      minIDx = i

  # print(cycles[minIDx])

  map_cycle_to_paths(corresponding_path_set, port_set, paths, ports_conn_matrix, switches, ports, cycles[0])

  # for i in range(len(path_set)):
  #   for j in range(i+1, len(path_set)):
  #     ret = deadlock_detection(path_set[i:j], switches, 16, ports_conn_matrix)
  #     if ret:
  #       print("i {} j {}".format(i, j))
  #       print("Deadlock Risk!!!")
  #       return

def test_edst_no_deadlock():
  mat_f = open('demo_mat', mode='r')
  lines = mat_f.read().splitlines()
  
  d_0 = lines[0].split(' ')

  host_per_sw = int(d_0[0])
  switches = int(d_0[1])
  links = int(d_0[2])
  total_hosts = int(d_0[3])
  ports_conn_matrix = {}
  for i in range(switches):
    ports_conn_matrix[i] = {}
    for j in range(switches):
      ports_conn_matrix[i][j] = []
  
  for i in range(1, links + 1):
    link = lines[i].split(' ')
    link = map(int, link)
    ports_conn_matrix[ link[0] ][ link[1] ] = [link[2], link[3]]
  
  path_f = open('path.txt', mode='r')
  paths = path_f.read().splitlines()
  d = map(int, paths[0].split(' '))
  pairs = d[0]
  idx = 1
  pas = []
  for _ in range(pairs):
    iii = map(int, paths[idx].split(' '))
    src = iii[0]
    dst = iii[1]
    pnum = iii[2]
    idx += 1
    for _ in range(pnum):
      p = map(int, paths[idx].split(' '))
      pas.append(p[1:])
      idx += 1
    
  # print(len(pas))
  ret, _, _, _ = deadlock_detection(pas, 64, 16, ports_conn_matrix)
  if ret : 
    print("deadlock")
  else:
    print("Not deadlock, you are lucky!")
  mat_f.close()

if __name__ == "__main__":
    # test_deadlock_detection1()
    # test_fc_ecmp_deadlock()
    test_edst_no_deadlock()