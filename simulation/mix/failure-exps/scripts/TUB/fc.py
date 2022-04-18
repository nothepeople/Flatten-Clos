import random
import time
import numpy as np
import os
import sys
import copy
import heapq
from multiprocessing import Process
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
  
  paths = []
  for dp in deduplicate_path:
    nodes = dp.replace('[', '').replace(']', '').split(',')
    path = []
    for node in nodes:
      path.append( int(node) )
    paths.append(path)
  return paths


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

def multi_core_routing_calculation(up_path_map, switches, layers, a, b):
  start_time = time.time()
  for i in range(a, b+1):
    for j in range(switches):
      calculation(up_path_map, switches, layers, i, j)
  end_time = time.time()
  print("Process {} running {}s.".format(a / 8, end_time - start_time))


def paths_of_any_pairs(vclos, layers, switches):
  up_path_map = virtual_up_down_routing(vclos, layers, switches)

  virtual_up_down_path = {}
  for i in range(switches):
    virtual_up_down_path[i] = {} 
    for j in range(switches):
      virtual_up_down_path[i][j] = []

  for i in range(switches):
    for j in range(i+1, switches):
      paths = calculation(up_path_map, switches, layers, i, j)
      virtual_up_down_path[i][j] = paths
  
  sum_paths = 0
  sum_path_len = 0
  sum_shortest_path_len = 0
  pairs = 0

  for i in range(switches):
    for j in range(i+1, switches):

      virtual_up_down_path[i][j] = sorted(virtual_up_down_path[i][j], key=lambda path : len(path))
      i_to_j_paths = virtual_up_down_path[i][j]
      
      ## PATH FEATURE
      sum_paths += len(i_to_j_paths)
      sum_shortest_path_len += len(i_to_j_paths[0])
      pairs += 1

      plen = 0
      j_to_i_paths = []
      for path in i_to_j_paths:
          plen += len(path)
          copyPath = copy.deepcopy(path)
          copyPath.reverse()
          j_to_i_paths.append(copyPath)
      sum_path_len += float(plen) / len(i_to_j_paths)
      virtual_up_down_path[j][i] = j_to_i_paths
  print("avg_paths: {:.2f}  avg_path_len: {:.2f} avg_shortest_path_len: {:.2f}".format(
    sum_paths / float(pairs),
    sum_path_len / float(pairs),
    sum_shortest_path_len / float(pairs)
  ))
  return virtual_up_down_path


def convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer):
  print("Construct Virtual Clos Network!")
  print(topo_matrix.shape)
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

  # print("Begin FC's wiring procedure!!!")
  bg_time = time.time()
  for layer in range(virtual_layers - 1):
    # print("Wiring layer {}".format(layer))
    for sA in range(switches):
      while switch_objs[sA].check_layer_i_has_unoccupied_port(layer):
        sB = random.randint(0, switches - 1)
        while not can_wire_between_switches(switch_objs[sA], switch_objs[sB], layer, layer+1, topo_matrix):
          cur_time = time.time()
          if cur_time - bg_time > 0.2:
            return False, None, None, None
          sB = random.randint(0, switches - 1)
        res, pA, pB = wiring(switch_objs[sA], switch_objs[sB], layer, layer+1)
        # print("Wire: res={} sA-{}, sB-{}, pA-{}, pB-{},".format(res, sA, sB, pA, pB))
        if res ==True:
          topo_matrix[sA][sB] = 1
          topo_matrix[sB][sA] = 1
          ports_conn_matrix[sA][sB].extend([pA, pB])
          ports_conn_matrix[sB][sA].extend([pB, pA])
  # print("FC's wiring procedure has completed!!!")
  return True, topo_matrix, ports_conn_matrix, switch_objs

def fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer):
  ret, topo_matrix, ports_conn_matrix, switch_objs = do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  retries = 0
  while not ret:
    retries += 1
    # print("Retry {} times".format(retries))
    ret, topo_matrix, ports_conn_matrix, switch_objs = do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  mat_file = open("demo_mat", mode='w')

  print("Write a mat file in topo-generator dir...")
  mat_file.write("%d %d %d %d\n" % (to_hosts, switches, (ports - to_hosts) * switches, switches * to_hosts) )
  for i in range(switches):
    for j in range(switches):
      if i==j or topo_matrix[i,j]==0: continue
      # print(ports_conn_matrix[i][j])
      mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
  mat_file.close()

  return topo_matrix, ports_conn_matrix, switch_objs


def topo_gen(switches, ports, to_hosts, ports_of_vir_layer,
            fc_ecmp_topo_dir="deadlock/topo/topo.txt",
            fc_ecmp_trace_dir="deadlock/trace/trace.txt"
  ):
  ret, topo_matrix, ports_conn_matrix, switch_objs = do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  retries = 0
  while not ret:
    retries += 1
    print("Retry {} times".format(retries))
    ret, topo_matrix, ports_conn_matrix, switch_objs = do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  
  mat_file = open("demo_mat", mode='w')
  print("Write a mat file in topo-gen function...")
  mat_file.write("%d %d %d %d\n" % (to_hosts, switches, (ports - to_hosts) * switches, switches * to_hosts) )
  for i in range(switches):
    for j in range(switches):
      if i==j or topo_matrix[i,j]==0: continue
      mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
  mat_file.close()

  hosts = switches * to_hosts
  host_ids = [id for id in range(hosts)]
  switch_ids = [id + hosts for id in range(switches)]

  ## fc-ecmp
  fc_ecmp_topo_file = open(fc_ecmp_topo_dir, mode='w')
  fc_ecmp_trace_file = open(fc_ecmp_trace_dir, mode='w')

  sw_to_sw_links = ports - to_hosts
  links = hosts + sw_to_sw_links * switches / 2
  total_nodes = hosts + switches
  
  print("total nodes: {} hosts: {} switches: {}".format(total_nodes, hosts, switches))
  
  fc_ecmp_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )
  fc_ecmp_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )
  for sw_idx in range(switches):
    for h in range(to_hosts):
      host_id = sw_idx * to_hosts + h
      sw_id = switch_ids[sw_idx]
      fc_ecmp_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )
  
  for sw1 in range(switches):
    for sw2 in range(sw1 + 1, switches ):
      if topo_matrix[sw1][sw2] == 1:
        ports = ports_conn_matrix[sw1][sw2]

        fc_ecmp_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[sw1], switch_ids[sw2], 1, 1000, ports[0], ports[1]) )
  fc_ecmp_topo_file.close()

  ## Write trace file
  fc_ecmp_trace_file.write("%d\n" % (total_nodes))
  fc_ecmp_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  fc_ecmp_trace_file.close()
  return topo_matrix, ports_conn_matrix, switch_objs

if __name__ == "__main__":
  # pass
  random.seed(12)
  ports_of_vir_layer = [3,7,7,3]
  layers=4
  switches=288
  ports=24
  to_hosts=4

  topo_matrix, ports_conn_matrix, switch_objs = topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
  