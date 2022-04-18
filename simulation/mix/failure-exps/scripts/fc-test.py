from ast import Pass
from distutils.dir_util import copy_tree
from itertools import permutations
import random
from re import S
from telnetlib import SB
import time
import numpy as np
import os
import sys
import copy
import heapq
from multiprocessing import Process, Value, Array
import os

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
  
  def mark_unused(self, port_id):
    self.used[port_id] = 0

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

  def print_port_usage_of_layer(self, layer):
    for port_id in self.port_ids[layer]:
      if self.used[port_id]:
        print("Used  ! switch: {} port_id: {} layer: {}".format(self.id, port_id, layer))
      else:
        print("Unused! switch: {} port_id: {} layer: {}".format(self.id, port_id, layer))

"""
VIRTUAL UP-DOWN
"""

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

def calculation(up_path_map, switches, layers, src, dst):
  paths = []
  for layer in range(layers - 1, layers):
    for i in range(switches):
      sId = i + layer * switches
      if up_path_map[sId][src] is not None and up_path_map[sId][dst] is not None:
        for p1 in up_path_map[sId][src]:
          for p2 in up_path_map[sId][dst]:
            path1 = copy.deepcopy(p1)
            path2 = copy.deepcopy(p2)
            path2.reverse()
            path1.extend(path2[1:])
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

  # print(paths)
  paths = sorted(paths, key=lambda path : len(path))
  avg_path_len /= float(len(paths))
  num_of_paths = len(paths)
  shortest_path_len = len(paths[0]) 
  # return paths, 0, 0, 0
  return paths, num_of_paths, avg_path_len, shortest_path_len


def multi_core_routing_calculation(up_path_map, virtual_up_down_path, 
    switches, 
    layers, 
    process_tasks,
    process_id, 
    fails,
    min_arr,
    path_num_arr, 
    path_len_arr, 
    shortest_path_len_arr):
  start_time = time.time()
  pairs = 0
  avg_num_paths = 0
  avg_path_len = 0
  avg_shortest_path_len = 0
  print("Process {} started".format(process_id))
  fail = 0
  min_paths = 0xfffff
  for source_dest_pair in process_tasks[process_id]:
    source = source_dest_pair[0]
    dest = source_dest_pair[1]
    paths, num_of_paths, _avg_path_len, shortest_path_len = calculation(up_path_map, switches, layers, source, dest)
    avg_num_paths += num_of_paths
    avg_path_len += _avg_path_len
    avg_shortest_path_len += shortest_path_len

    pairs += 1
    # print(paths[0])
    virtual_up_down_path[source][dest] = paths
    # print(len(virtual_up_down_path[source][dest]))
    if len(paths) < min_paths:
      min_paths = len(paths)
    if len(paths) == 0:
        fail += 1
  fails.value += fail
  end_time = time.time()

  avg_num_paths /= float(pairs)
  avg_path_len /= float(pairs)
  avg_shortest_path_len /= float(pairs)
  min_arr[process_id] = min_paths
  path_num_arr[process_id] = avg_num_paths
  path_len_arr[process_id] = avg_path_len
  shortest_path_len_arr[process_id] = shortest_path_len
  
  print("Process {} run@{:.2f}s".format(process_id, end_time - start_time))


def paths_of_any_pairs(vclos, layers, switches, num_process):
  up_path_map = virtual_up_down_routing(vclos, layers, switches)

  virtual_up_down_path = {}
  for i in range(switches):
    virtual_up_down_path[i] = {} 
    for j in range(switches):
      virtual_up_down_path[i][j] = []

  source_des_pairs = []
  for i in range(switches):
    for j in range(i + 1, switches):
      source_des_pairs.append([i, j])
  total_tasks = len(source_des_pairs)

  process_tasks = {}
  for i in range(num_process):
    process_tasks[i] = []
  
  ## Split tasks
  # print(total_tasks)
  ids = [0]
  for i in range(1, num_process):
    ids.append(i * (total_tasks // num_process))
  if ids[len(ids) - 1] < total_tasks: ids.append(total_tasks)
  # print(ids)
  for i in range(num_process):
    # /print(i)
    for j in range(ids[i], ids[i + 1]):
      # print(source_des_pairs[j])
      # time.sleep(1)
      process_tasks[i].append(source_des_pairs[j])
  s = 0
  for i in range(num_process):
    s += len(process_tasks[i])
  print(s, total_tasks)

  processes = []
  fails = Value('i', 0)
  path_num_arr = Array('d', range(num_process))
  shortest_path_len_arr = Array('d', range(num_process))
  path_len_arr = Array('d', range(num_process))
  min_arr = Array('d', range(num_process))

  for process_id in range(num_process):
    p = Process(target=multi_core_routing_calculation, args=(up_path_map, 
        virtual_up_down_path,
        switches, 
        layers, 
        process_tasks,
        process_id,
        fails,
        min_arr,
        path_num_arr,
        path_len_arr,
        shortest_path_len_arr
    ))
    processes.append(p)

  for process in processes:
    process.start()
  for process in processes:
    process.join()
  print("Avg_num_of_paths: {:.2f}, avg_path_len: {:.2f}, avg_shortest_path_len: {:.2f}  fail count: {} minimum_paths: {}".format(np.average(path_num_arr), np.average(path_len_arr), np.average(shortest_path_len_arr), fails.value, np.min(min_arr)))
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

def wiring(sA, sB, lA, lB, topo_matrix, ports_conn_matrix, switch_objs):
  # print("sA:{} unused:{} sB:{} unused:{}".format(sA, switch_objs[sA].count_layer_i_unused_ports(lA), sB, switch_objs[sB].count_layer_i_unused_ports(lB)))
  pA = switch_objs[sA].get_layer_i_unoccupied_port(lA)
  pB = switch_objs[sB].get_layer_i_unoccupied_port(lB)
  # print("pA= {} pB={}".format(pA, pB))
  if pA != -1 and pB != -1:
    switch_objs[sA].mark_used(pA)
    switch_objs[sB].mark_used(pB)
    topo_matrix[sA][sB] = 1
    topo_matrix[sB][sA] = 1
    ports_conn_matrix[sA][sB].extend([pA, pB])
    ports_conn_matrix[sB][sA].extend([pB, pA])
    return True
  return False

def disconnect(sA, sB, lA, topo_matrix, ports_conn_matrix, switch_objs):
  conn_ports = ports_conn_matrix[sA][sB]
  pA = conn_ports[0]
  pB = conn_ports[1]
  ports_conn_matrix[sA][sB] = []
  ports_conn_matrix[sB][sA] = []
  topo_matrix[sA][sB] = 0
  topo_matrix[sB][sA] = 0
  switch_objs[sA].mark_unused(pA)
  switch_objs[sB].mark_unused(pB)

"""
need to readjust some links
？
1SA, S 2,，1,）
2）SA, S（S）
3）SA, SS
"""

def readjust_links(sA, lA, switches, topo_matrix, ports_conn_matrix, switch_objs):
  """
  sA is in the lower layer, sB is in the upper layer, lA + 1 = lB
  can not establish connection between sA and sB, because:
  1) sB don't have enough ports of layer lB to be connected
  2) sB have enough ports but sB has connection with sA at the lower (< lA) layer
  """
  need_rewired = []
  sA_unused_ports = switch_objs[sA].count_layer_i_unused_ports(lA)
  # print("sA: {} Unused port: {} @layer: {}".format(sA, sA_unused_ports, lA))
  sDs = []
  if sA_unused_ports < 2:
    for _ in range(2 - sA_unused_ports):
      for sD in range(switches):
        if topo_matrix[sA][sD] == 0: continue
        ## sA has one connection with sD
        pA = ports_conn_matrix[sA][sD][0]
        pD = ports_conn_matrix[sA][sD][1]
        
        ## disconnect sA with sD at (lA + 1)-layer
        mapped_layer_of_sA = switch_objs[sA].map_port_to_virtual_layers(pA)
        mapped_layer_of_sD= switch_objs[sD].map_port_to_virtual_layers(pD)
        if topo_matrix[sA][sD] == 1 and mapped_layer_of_sA == lA and mapped_layer_of_sD == lA + 1:
          # print("before dis-->sD: {} Unused port: {} @layer: {}".format(sD, switch_objs[sD].count_layer_i_unused_ports(lA + 1), lA + 1))
          disconnect(sA, sD, lA, topo_matrix, ports_conn_matrix, switch_objs)
          # print("After dis-->sD: {} Unused port: {} @layer: {}".format(sD, switch_objs[sD].count_layer_i_unused_ports(lA + 1), lA + 1))
          sDs.append(sD)
          break
  
  # print("disconnect=", sDs)
  
  ## sA has two unused ports
  if len(sDs) == 1:
    # print("=============> len SD 1 <================")
    sD = sDs[0]
    for sB in range(switches):
      ## choose one sB has no connection with sA
      if sB == sA or topo_matrix[sA][sB] == 1: continue
      
      ## get sB's connections with other switches at layer 'lA'.
      for sC in range(switches):
        # print(sB, sC)
        if topo_matrix[sB][sC] == 0: continue 
        conn_ports = ports_conn_matrix[sB][sC] 
        pB = conn_ports[0]
        pC = conn_ports[1]
        mapped_layer_of_sB = switch_objs[sB].map_port_to_virtual_layers(pB)
        mapped_layer_of_sC= switch_objs[sC].map_port_to_virtual_layers(pC)
        sC_unused_ports_at_lA_plus_1 = switch_objs[sC].count_layer_i_unused_ports(lA + 1)
        if topo_matrix[sC][sD] == 0 and sC != sD and topo_matrix[sA][sC] == 0 and sC_unused_ports_at_lA_plus_1 > 0 and mapped_layer_of_sC == lA and mapped_layer_of_sB == lA + 1:
          # print("disconnect")
          disconnect(sB, sC, lA + 1, topo_matrix, ports_conn_matrix, switch_objs)
          # disconnect sB with sC
          # print("wring sA: {} sB: {}".format(sA, sB))
          res = wiring(sA, sB, lA, lA+1, topo_matrix, ports_conn_matrix, switch_objs)
          # if res ==True:
            # print("wring sA: {} sB: {} successfully\n".format(sA, sB))
          
          # print("wring sA: {} sC: {}".format(sA, sC))
          ret = wiring(sA, sC, lA, lA+1, topo_matrix, ports_conn_matrix, switch_objs)
          # if res ==True:
            # print("wring sA: {} sC: {} successfully\n".format(sA, sC))
          
          # print("wring sC: {} sD: {}".format(sC, sD))
          res = wiring(sC, sD, lA, lA+1, topo_matrix, ports_conn_matrix, switch_objs)
          # if res ==True:
          #   print("wring sA: {} sC: {} successfully\n".format(sC, sD))
            
          # print("before len1 ret sA: {} Unused port: {} @layer: {}".format(sA, switch_objs[sA].count_layer_i_unused_ports(lA), lA))
          return True, None
  elif len(sDs) == 0:
    # print("============> LEN SD 00 <==============")
    for sB in range(switches):
      ## choose one sB has no connection with sA
      if sB == sA or topo_matrix[sA][sB] == 1: continue
      for sC in range(switches):
        if topo_matrix[sB][sC] == 0: continue 
        conn_ports = ports_conn_matrix[sB][sC] 
        pB = conn_ports[0]
        pC = conn_ports[1]
        mapped_layer_of_sB = switch_objs[sB].map_port_to_virtual_layers(pB)
        mapped_layer_of_sC= switch_objs[sC].map_port_to_virtual_layers(pC)
        # time.sleep(0.5)
        # print(sB, sC)
        if mapped_layer_of_sB == lA + 1 and mapped_layer_of_sC == lA:
          # print("len 0000 dis connect")
          disconnect(sB, sC, lA, topo_matrix, ports_conn_matrix, switch_objs)
          # disconnect sB with sC
          res = wiring(sA, sB, lA, lA+1, topo_matrix, ports_conn_matrix, switch_objs)
          # if res ==True:
          #   print("wring sA: {} sB: {} successfully----0000\n".format(sA, sB))
          
          # res = wiring(sA, sC, lA, lA+1, topo_matrix, ports_conn_matrix, switch_objs)
          # if res ==True:
          #   print("wring sA: {} sC: {} successfully----0000\n".format(sA, sC))
          
          # print("000000 ret sA: {} Unused port: {} @layer: {}".format(sA, switch_objs[sA].count_layer_i_unused_ports(lA), lA))
          
          ## sC has one port need to be wired
          need_rewired.append(sC)
          return True, need_rewired
  return False, None


def do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer):
  # print("call do_topo_gen")
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

  # print("Begin FC's wiring procedure!!!")
  for layer in range(virtual_layers - 1):
    print("Wiring layer {}".format(layer))
    rewired = []
    for sA in range(switches):
      
      while switch_objs[sA].check_layer_i_has_unoccupied_port(layer):
        permutations = np.random.permutation(switches)
        idx = 0
        sB = -1
        while idx < switches:
          sB = permutations[idx]
          if not can_wire_between_switches(switch_objs[sA], switch_objs[sB], layer, layer+1, topo_matrix):
            idx += 1
          else:
            break
        
        if idx >= switches and not can_wire_between_switches(switch_objs[sA], switch_objs[sB], layer, layer+1, topo_matrix):
          while switch_objs[sA].check_layer_i_has_unoccupied_port(layer):
            # time.sleep(1)
            _, need_wired, = readjust_links(sA, layer, switches, topo_matrix, ports_conn_matrix, switch_objs)
            if need_wired is not None: rewired.extend(need_wired)
          # print("readjust over...")
          continue
         
        res = wiring(sA, sB, layer, layer+1, topo_matrix, ports_conn_matrix, switch_objs)
        if res ==True:
          pass
    # print("layer: {} rewire switches: {}".format(layer, len(rewired)))
    while len(rewired) > 0:
      rewire_sw = rewired[0]
      while switch_objs[rewire_sw].check_layer_i_has_unoccupied_port(layer):
        _, need_wired, = readjust_links(rewire_sw, layer, switches, topo_matrix, ports_conn_matrix, switch_objs)
        if need_wired is not None: rewired.extend(need_wired)
      # print('reamin=', len(rewired))
      rewired.remove(rewire_sw)
  links = 0
  for i in range(switches):
    for j in range(switches):
      if i >= j and topo_matrix[i][j] == 1: links += 1
  print("links={}".format(links))
  return True, topo_matrix, ports_conn_matrix, switch_objs

def fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer):
  
  ret, topo_matrix, ports_conn_matrix, switch_objs = do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  print("topo_ret=", ret)
  return topo_matrix, ports_conn_matrix, switch_objs

def switch_500_test():
  switches=500
  ports=64
  to_hosts=24
  
  ports_of_vir_layer=[10, 20, 10]
  layers= len (ports_of_vir_layer)
  
  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)

  ports_of_vir_layer=[7, 13, 13, 7]
  layers= len (ports_of_vir_layer)

  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)
  
  ports_of_vir_layer=[5, 10, 10, 10, 5]
  layers= len (ports_of_vir_layer)
  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)


def switch_1000_test():
  switches=1000
  ports=64
  to_hosts=24
  
  ports_of_vir_layer=[10, 20, 10]
  layers= len (ports_of_vir_layer)
  
  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)

  ports_of_vir_layer=[7, 13, 13, 7]
  layers= len (ports_of_vir_layer)

  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)
  
  ports_of_vir_layer=[5, 10, 10, 10, 5]
  layers= len (ports_of_vir_layer)
  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)


def switch_5000_test():
  switches=1000
  ports=64
  to_hosts=24
  ports_of_vir_layer=[5, 10, 10, 10, 5]
  
  layers= len (ports_of_vir_layer)
  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)
  
  switches=5000
  ports=64
  to_hosts=24
  
  ports_of_vir_layer=[7, 13, 13, 7]
  layers= len (ports_of_vir_layer)
  
  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)

  ports_of_vir_layer=[5, 10, 10, 10, 5]
  layers= len (ports_of_vir_layer)

  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)
  
  ports_of_vir_layer=[4, 8, 8, 8, 8, 4]
  layers= len (ports_of_vir_layer)
  topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  _ = paths_of_any_pairs(vclos, layers, switches, num_process)


if __name__ == "__main__":
  # num of process
  num_process = 8
  # switch_500_test()
  # switch_500_test()
  # switch_1000_test()
  switch_5000_test()
  # a = [3,5,8,9]
  # print(random.sample(a, 2))
  # random.seed(777931)
  # ports_of_vir_layer=[4, 8, 8, 4]
  # layers=4
  # switches=1024
  # ports=32
  # to_hosts=8

  # ports_of_vir_layer=[3, 6, 3]
  # layers=3
  # switches=32
  # ports=12
  # to_hosts=0

  # random.seed(12)
  
  # ports_of_vir_layer=[7, 13, 13, 7]
  # layers= len (ports_of_vir_layer)
  # switches=5000
  # ports=64
  # to_hosts=24

  # # while True:
  # topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  # vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
  # fails = paths_of_any_pairs(vclos, layers, switches, num_process)
