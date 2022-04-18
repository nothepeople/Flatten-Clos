import random
import time
import numpy as np
import os
import sys
import copy
import heapq
from multiprocessing import Process, Value, Array
import os

from ecmp import cal_shortest_path
from deadlock_detection import deadlock_detection, map_cycle_to_paths

class Switch:
  def __init__(self, id, ports=16, to_hosts=4, layers=3, ports_of_vir_layer=[4,6,2]):
    self.id = id
    self.used = [0 for _ in range(ports)]
    self.to_hosts = to_hosts
    # check param setting
    if layers != len(ports_of_vir_layer) or np.sum(ports_of_vir_layer) + to_hosts != ports:
      print("param error")
      sys.exit(-1)
    self.layers = layers
    self.ports_of_vir_layer = ports_of_vir_layer
    self.port_ids = {}
    pre = 0
    for i in range(layers):
      self.port_ids[i] = [to_hosts + pre + ii for ii in range(ports_of_vir_layer[i])]
      pre = np.sum(ports_of_vir_layer[:i+1])

    # print(self.port_ids)

    ## FC has layer - 1 bipartite graphs
    self.degrees = []
    for i in range(layers):
      degree = 0
      if i == 0:
        degree = ports_of_vir_layer[0]
      else:
        degree = ports_of_vir_layer[i] - self.degrees[len(self.degrees) - 1]
      self.degrees.append(degree)


  def map_port_to_virtual_layers(self, port_id):
    for layer in range(self.layers):
      if port_id in self.port_ids[layer]:
        return layer
    return -1

  def count_layer_i_used_ports(self, layer_i):
    count = 0
    for port_id in self.port_ids[layer_i]:
      if self.used[port_id]:
        count += 1
    return count
  
  def count_layer_i_unused_ports(self, layer_i):
    count = 0
    for port_id in self.port_ids[layer_i]:
      if not self.used[port_id]:
        count += 1
    return count

  def get_vir_layer_i_ports(self, layer_i):
    return self.ports_of_vir_layer[layer_i]

  def mark_used(self, port_id):
    self.used[port_id] = 1

  def check_layer_i_has_unoccupied_port(self, layer):
    has = False
    for port in self.port_ids[layer]:
      if not self.used[port]:
        has = True
        return has
    return has

  def get_layer_i_unoccupied_port(self, layer):
    rnd = random.randint(0, len(self.port_ids[layer])-1)
    selected_port_no = self.port_ids[layer][rnd]
    while self.used[selected_port_no]:
      rnd = random.randint(0, len(self.port_ids[layer])-1)
      selected_port_no = self.port_ids[layer][rnd]
    # print(selected_port_no)
    return selected_port_no


def calculation(up_path_map, switches, layers, src, dst):
  paths = []
  for layer in range(1, layers):
    for i in range(switches):
      sId = i + layer * switches
      if up_path_map[sId][src] is not None and up_path_map[sId][dst] is not None:
        for p1 in up_path_map[sId][src]:
          for p2 in up_path_map[sId][dst]:
            path1 = copy.deepcopy(p1)
            path2 = copy.deepcopy(p2)
            path2.reverse()
            path1.extend(path2[1:])
            # print(path1)
            phy_path = []
            for vnode in path1:
              if len(phy_path) == 0:
                phy_path.append(vnode % switches)
                continue
              if vnode % switches == phy_path[len(phy_path) - 1]:
                continue
              phy_path.append(vnode % switches)
            # print("{}->{}: paths:{}".format(src, dst, len(phy_path)))
            paths.append(phy_path)
  deduplicate_path = set([])
  for path in paths:
    deduplicate_path.add(str(path))
  
  # deduplicate and calculate path features
  paths = []
  num_of_paths = 0.0
  shortest_path_len = 0.0
  avg_path_len = 0.0
  for dp in deduplicate_path:
    nodes = dp.replace('[', '').replace(']', '').split(',')
    path = []
    for node in nodes:
      path.append( int(node) )
    
    avg_path_len += len(path)
    paths.append(path)

  # print("{}->{} num of paths: {} paths: {}".format(src, dst, len(paths), paths))
  paths = sorted(paths, key=lambda path : len(path))
  avg_path_len /= float(len(paths))
  num_of_paths = len(paths)
  shortest_path_len = len(paths[0]) 

  return paths, num_of_paths, avg_path_len, shortest_path_len


def virtual_up_down_routing(vclos, layers, switches):
  if vclos is None:
    print("param @vclos should not be None")
    return
  num_vir_switches = vclos.shape[0]
 
  up_path_map = {}
  for i in range(num_vir_switches): # for each virtual node
    up_path_map[i] = {}
    for j in range(switches): # bottom layer switches
      up_path_map[i][j] = []

  for i in range(switches):
    up_path_map[i][i].append([i])
  
  """
  each S_i_1 -> S_i_j
  """
  print("layers: {}".format(layers))
  for layer in range(1, layers):
    ## uppper layer switch
    for idx1 in range(switches):
      s1 = idx1 + layer * switches
      ## bottom layer switch
      for idx2 in range(switches):
        s2 = idx2 + (layer - 1) * switches
        # print("layer: {} s1: {} s2: {} idx1:{} idx2:{}".format(layer, s1, s2, idx1, idx2))
        if vclos[s1][s2] == 1:
          ## the layer-1 virtual switch
          for s3 in range(switches):
            if len(up_path_map[s2][s3]) == 0:
              continue
            # print(len(up_path_map[s2][s3]))
            for p in up_path_map[s2][s3]:
              ## need deepcopy
              deepP = copy.deepcopy(p)
              deepP.append(s1)
              # print(p)
              up_path_map[s1][s3].append(deepP)
  return up_path_map

INF = 1204

def multi_core_routing_calculation(up_path_map, 
    switches, 
    layers, 
    cores, 
    core_id, 
    fails, 
    path_num_arr, 
    path_len_arr, 
    shortest_path_len_arr, 
    a, b):
  start_time = time.time()
  pairs = 0
  avg_num_paths = 0
  avg_path_len = 0
  avg_shortest_path_len = 0
  # print("Process {} start@{}s cal:({}, {})".format(core_id, start_time, a, b))
  fail = 0
  for i in range(a, b):
    for j in range(switches):
      paths, num_of_paths, _avg_path_len, shortest_path_len = calculation(up_path_map, switches, layers, i, j)
      
      avg_num_paths += num_of_paths
      avg_path_len += _avg_path_len
      avg_shortest_path_len += shortest_path_len

      pairs += 1
      # print(paths[0])
      if len(paths) == 0:
          fail += 1
  fails.value += fail
  end_time = time.time()

  avg_num_paths /= float(pairs)
  avg_path_len /= float(pairs)
  avg_shortest_path_len /= float(pairs)

  path_num_arr[core_id] = avg_num_paths
  path_len_arr[core_id] = avg_path_len
  shortest_path_len_arr[core_id] = shortest_path_len
  
  print("Process {} run@{:.2f}s cal:({}, {})".format(core_id, end_time - start_time, a, b))


def paths_of_any_pairs(vclos, layers, switches, num_process):
  up_path_map = virtual_up_down_routing(vclos, layers, switches)

  virtual_up_down_path = {}
  for i in range(switches):
    virtual_up_down_path[i] = {} 
    for j in range(switches):
      virtual_up_down_path[i][j] = []


  up_path_maps = []
  
  for i in range(num_process):
    copied_map = copy.deepcopy(up_path_map)
    up_path_maps.append(copied_map)
  
  processes = []
  fails = Value('i', 0)
  path_num_arr = Array('d', range(num_process))
  shortest_path_len_arr = Array('d', range(num_process))
  path_len_arr = Array('d', range(num_process))

  for i in range(num_process):
    p = Process(target=multi_core_routing_calculation, args=(up_path_maps[i], 
        switches, 
        layers, 
        num_process,
        i,
        fails,
        path_num_arr,
        path_len_arr,
        shortest_path_len_arr,
        i * (switches / num_process), 
        (i + 1) * (switches / num_process)
    ))
    processes.append(p)

  for process in processes:
    process.start()
  for process in processes:
    process.join()
  print("Avg_num_of_paths: {:.2f}, avg_path_len: {:.2f}, avg_shortest_path_len: {:.2f}  fail count: {}".format(np.average(path_num_arr), np.average(path_len_arr), np.average(shortest_path_len_arr), fails.value))
  # print("process over, fails{}".format(fails))
  return fails


def convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer):
  print("Construct Virtual Clos Network!")
  # /print(topo_matrix.shape)
  switches = topo_matrix.shape[0]
  layers = len(ports_of_vir_layer)
  vclos = np.zeros(shape=(layers * switches, layers * switches))
  # INF = float('inf')
  # for i in range(virtual_layers - 1):
  for i in range(switches):
    vclos[i][i] = INF
    for j in range(1, layers):
      vclos[i + j * switches][i + j * switches] = INF
      vclos[i + (j - 1) * switches][i + j * switches] = 1
      vclos[i + j * switches][i + (j - 1) * switches] = 1

  for sA in range(switches):
    for sB in range(switches):
      if topo_matrix[sA][sB] == 0:
        continue
      ports = ports_conn_matrix[sA][sB]
      pA = ports[0]
      pB = ports[1]
      lA = switch_objs[sA].map_port_to_virtual_layers(pA)
      lB = switch_objs[sB].map_port_to_virtual_layers(pB)
      ## A is a lower layer switch or B is a lower layer switch
      vclos[sA + switches * lA][sB + switches * lB] = 1
      vclos[sB + switches * lB][sA + switches * lA] = 1
  return vclos
  paths_of_any_pairs(vclos, layers, switches)
  # print(vclos)
  # virtual_up_down_routing(vclos, layers, switches)

"""
functions of topo generation
"""
def can_wire_between_switches(sA, sB, lA, lB, topo_matrix):
  if sA.id == sB.id:
    return False
  if topo_matrix[sA.id][sB.id] == 1:
    return False
  used_ports_of_sA_at_lA = sA.count_layer_i_used_ports(lA)
  used_ports_of_sB_at_lB = sB.count_layer_i_used_ports(lB)
  
  return sA.degrees[lA] - used_ports_of_sB_at_lB > 0

def wiring(swA, swB, lA, lB):
  pA = swA.get_layer_i_unoccupied_port(lA)
  pB = swB.get_layer_i_unoccupied_port(lB)
  if pA != -1 and pB != -1:
    swA.mark_used(pA)
    swB.mark_used(pB)
    return True, pA, pB
  return False, -1, -1

def do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer):
  virtual_layers = len(ports_of_vir_layer)
  switch_objs = []
  switch_ids = [id for id in range(switches)]
  for i in range(switches):
    switch_objs.append( Switch(id=switch_ids[i], ports=ports, to_hosts=to_hosts, layers=len(ports_of_vir_layer), ports_of_vir_layer=ports_of_vir_layer) )
  
  topo_matrix = np.zeros(shape=(switches, switches), dtype=int)
  ports_conn_matrix = {}
  for i in range(switches):
    ports_conn_matrix[i] = {}
    for j in range(switches):
      ports_conn_matrix[i][j] = []

  print("Begin FC's wiring procedure!!!")
  
  for layer in range(virtual_layers - 1):
    print("Wiring layer {}".format(layer))
    for sA in range(switches):
      # print("sA:{}".format(sA))
      while switch_objs[sA].check_layer_i_has_unoccupied_port(layer):
        sB = random.randint(0, switches - 1)
        bg_time = time.time()
        while not can_wire_between_switches(switch_objs[sA], switch_objs[sB], layer, layer+1, topo_matrix):
          cur_time = time.time()
          if cur_time - bg_time > 0.8:
            return False, None, None, None
          sB = random.randint(0, switches - 1)
        res, pA, pB = wiring(switch_objs[sA], switch_objs[sB], layer, layer+1)
        # print("Wire: res={} sA-{}, sB-{}, pA-{}, pB-{},".format(res, sA, sB, pA, pB))
        if res ==True:
          topo_matrix[sA][sB] = 1
          topo_matrix[sB][sA] = 1
          ports_conn_matrix[sA][sB].extend([pA, pB])
          ports_conn_matrix[sB][sA].extend([pB, pA])
  print("FC's wiring procedure has completed!!!")
  return True, topo_matrix, ports_conn_matrix, switch_objs

def fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer):
  
  ret, topo_matrix, ports_conn_matrix, switch_objs = do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  retries = 0
  sed = 6
  while not ret:
    retries += 1
    # sd = random.randint(0, 1000000)
    random.seed(sed)
    print("Retry {} times, seed {}".format(retries, sed))
    sed += 1
    ret, topo_matrix, ports_conn_matrix, switch_objs = do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  return topo_matrix, ports_conn_matrix, switch_objs

def get_shortest_path_list(topo_matrix, switches):
    sp_list = []
    shortest_path_pair = cal_shortest_path(topo_matrix, switches)
    for src in range(switches):
        for dst in range(switches):
            if src == dst: continue
            for path in shortest_path_pair[src][dst]:
                sp_list.append(path)
    return sp_list

def test_practical_deadlock():

  switches = 4
  ports = 4

  topo_matrix = np.zeros((4,4))

  topo_matrix[0][1] = 1
  topo_matrix[0][2] = 1
  topo_matrix[0][3] = 1
  topo_matrix[1][2] = 1
  topo_matrix[1][3] = 1

  topo_matrix[1][0] = 1
  topo_matrix[2][0] = 1
  topo_matrix[3][0] = 1
  topo_matrix[2][1] = 1
  topo_matrix[3][1] = 1

  ports_conn_matrix = {}
  for i in range(switches):
    ports_conn_matrix[i] = {}
    for j in range(switches):
      ports_conn_matrix[i][j] = None
  
  ports_conn_matrix[0][1] = [1, 0]
  ports_conn_matrix[1][0] = [0, 1]
  
  ports_conn_matrix[0][2] = [3,1]
  ports_conn_matrix[2][0] = [1, 3]

  ports_conn_matrix[0][3] = [0, 1]
  ports_conn_matrix[3][0] = [1, 0]

  ports_conn_matrix[1][2] = [2, 0]
  ports_conn_matrix[2][1] = [0, 2]
  
  ports_conn_matrix[1][3] = [1, 3]
  ports_conn_matrix[3][1] = [3, 1]

  shortest_paths = get_shortest_path_list(topo_matrix, switches)
  has_cycle, _, _, _ = deadlock_detection(shortest_paths, switches, ports, ports_conn_matrix)
  print(has_cycle)
  if has_cycle:
      print("Has cycle in ECMP paths")
  

if __name__ == "__main__":

  test_practical_deadlock()
  """
  # num of process
  num_process = 1

  
  ports_of_vir_layer=[2,2]
  layers=len(ports_of_vir_layer)
  switches=4
  ports=4
  to_hosts=0

  random.seed(12)
  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)

#   for i in range(switches):
#     for j in range(i + 1, switches):
#       if topo_matrix[i][j] == 0: continue
#       print("{}->{} conn ports: {}".format(i, j, ports_conn_matrix[i][j]))
#   vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
# #   print("vclos: {}".format(vclos))
#   vcs = np.array(vclos, dtype=int)
#   np.savetxt("practical_vclos", vcs, fmt="%d")
#   fails = paths_of_any_pairs(vclos, layers, switches, num_process)

  shortest_paths = get_shortest_path_list(topo_matrix, switches)
  has_cycle, cycles, port_set, corresponding_path_set = deadlock_detection(shortest_paths, switches, ports, ports_conn_matrix)
  if has_cycle:
      print("Has cycle in ECMP paths")
  map_cycle_to_paths(corresponding_path_set, port_set, shortest_paths, ports_conn_matrix, switches, ports, cycles[0])
  """
