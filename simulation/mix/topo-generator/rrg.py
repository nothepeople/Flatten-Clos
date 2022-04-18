import random
import time
import numpy as np
import os
# from datetime import datetime

random.seed(1001)
class Switch:
  def __init__(self, id, ports, to_hosts, lefts=4, mid=6, right=2):
    self.id = id
    self.used = [0 for _ in range(ports)]
    self.to_hosts = to_hosts

    self.left_part_ports = [to_hosts + i for i in range(lefts)]
    self.mid_part_ports = [to_hosts + lefts + i for i in range(mid)]
    self.right_part_ports = [to_hosts + mid + lefts + i for i in range(right)]

  def mark_used(self, port_id):
    self.used[port_id] = 1

  def get_left_unoccupied_port(self):
    for left in self.left_part_ports:
      if not self.used[left]:
        return left
    return -1
  
  def get_mid_unoccupied_port(self):
    for mid in self.mid_part_ports:
      if not self.used[mid]:
        return mid
    return -1
  
  
  def get_right_unoccupied_port(self):
    for right in self.right_part_ports:
      if not self.used[right]:
        return right
    return -1
  
  def has_unoccupied_port(self):
    
    for u in self.used:
      if u == 0:
        return True
    return False
  
  def switch_level_port_range(self):
    return "{} {} {} {} {} {}" .format(
      self.left_part_ports[0], 
      self.left_part_ports[len(self.left_part_ports) - 1], 
      self.mid_part_ports[0], 
      self.mid_part_ports[len(self.mid_part_ports) - 1], 
      self.right_part_ports[0], 
      self.right_part_ports[len(self.right_part_ports) - 1])


  def __str__(self):
    return "sw_id={} used={} left_part_ports={} mid_part_ports={} right_part_ports={}".format(self.id, self.used, self.left_part_ports, self.mid_part_ports, self.right_part_ports)

def switch_has_empty_port(switch_objs):
  for sw in switch_objs:
    if sw.has_unoccupied_port() == True:
      return True
  return False

def connect_switches(swA, swB):
  if not swA.has_unoccupied_port() or not swB.has_unoccupied_port():
    return False, -1, -1
  p = random.random()
  if p < 0.25:
    p1 = swA.get_left_unoccupied_port()
    p2 = swB.get_mid_unoccupied_port()
    if p1 == -1 or p2 == -1:
      return False, -1, -1
    swA.mark_used(p1)
    swB.mark_used(p2)
    return True, p1, p2
  elif 0.25 <= p < 0.75 : # switchA mid port
    p1 = swA.get_mid_unoccupied_port()
    p2 = swB.get_left_unoccupied_port() if random.random() < 0.5 else swB.get_right_unoccupied_port()
    if p1 == -1 or p2 == -1:
      return False, -1, -1
    swA.mark_used(p1)
    swB.mark_used(p2)
    return True, p1, p2
  elif p >= 0.75:
    p1 = swA.get_right_unoccupied_port()
    p2 = swB.get_mid_unoccupied_port()
    if p1 == -1 or p2 == -1:
      return False, -1, -1
    swA.mark_used(p1)
    swB.mark_used(p2)
    return True, p1, p2


def has_links(link_matrix):
  sws = len(link_matrix)
  for i in range(sws):
    for j in range(sws):
      if link_matrix[i][j] == 0:
        return True
  return False


def rrg(hosts, switches, host_per_switch=1, ports=1,  host_bandwidth=1, 
        rrg_topo_dir="rrg-topo.txt", 
        rrg_mat_dir="rrg-mat.txt", 
        ksp_mat_dir="../../ksp/ksp_mat",
        rrg_trace_dir="rrg-trace.txt",

        rrg_ecmp_topo_dir="rrg-ecmp-topo.txt",
        rrg_ecmp_trace_dir="rrg-ecmp-trace",

        rrg_uniform_topo_dir="rrg-uniform-topo.txt",
        rrg_uniform_mat_dir="rrg-uniform-mat.txt",
        rrg_uniform_trace_dir="rrg-uniform-trace",

        rrg_weighted_topo_dir="rrg-weighted-topo.txt",
        rrg_weighted_mat_dir="rrg-weighted-mat.txt",
        rrg_weighted_trace_dir="rrg-weighted-trace",

        rrg_lp_topo_dir="rrg-lp-topo.txt",
        rrg_lp_mat_dir="rrg-lp-mat.txt",
        rrg_lp_trace_dir="rrg-lp-trace",

        edst_topo_dir="edst-topo.txt", 
        edst_mat_dir="edst-mat.txt", 
        edst_trace_dir="edst-trace.txt",

        edst_uniform_topo_dir="edst-uniform-topo.txt",
        edst_uniform_mat_dir="edst-uniform-mat.txt",
        edst_uniform_trace_dir="edst-uniform--trace",

        edst_weighted_topo_dir="edst-weighted-topo.txt",
        edst_weighted_mat_dir="edst-weighted-mat.txt",
        edst_weighted_trace_dir="edst-weighted-trace",

        edst_lp_topo_dir="edst-lp-topo.txt",
        edst_lp_mat_dir="edst-lp-mat.txt",
        edst_lp_trace_dir="edst-lp-trace",

        k_clos=3):
  host_ids = [id for id in range(hosts)]
  switch_ids = [id + hosts for id in range(switches)]
  ## rrg
  rrg_topo_file = open(rrg_topo_dir, mode='w')
  rrg_mat_file = open(rrg_mat_dir, mode='w')
  rrg_trace_file = open(rrg_trace_dir, mode='w')
  ksp_mat_file = open(ksp_mat_dir, mode='w')
  ## rrg-ecmp
  rrg_ecmp_topo_file = open(rrg_ecmp_topo_dir, mode='w')
  rrg_ecmp_trace_file = open(rrg_ecmp_trace_dir, mode='w')

  ## rrg-weighted
  rrg_uniform_topo_file = open(rrg_uniform_topo_dir, mode='w')
  rrg_uniform_mat_file = open(rrg_uniform_mat_dir, mode='w')
  rrg_uniform_trace_file = open(rrg_uniform_trace_dir, mode='w')

  rrg_weighted_topo_file = open(rrg_weighted_topo_dir, mode='w')
  rrg_weighted_mat_file = open(rrg_weighted_mat_dir, mode='w')
  rrg_weighted_trace_file = open(rrg_weighted_trace_dir, mode='w')

  rrg_lp_topo_file = open(rrg_lp_topo_dir, mode='w')
  rrg_lp_mat_file = open(rrg_lp_mat_dir, mode='w')
  rrg_lp_trace_file = open(rrg_lp_trace_dir, mode='w')

  edst_topo_file = open(edst_topo_dir, mode='w')
  edst_mat_file = open(edst_mat_dir, mode='w')
  edst_trace_file = open(edst_trace_dir, mode='w')

  edst_uniform_topo_file = open(edst_uniform_topo_dir, mode='w')
  edst_uniform_mat_file = open(edst_uniform_mat_dir, mode='w')
  edst_uniform_trace_file = open(edst_uniform_trace_dir, mode='w')

  edst_weighted_topo_file = open(edst_weighted_topo_dir, mode='w')
  edst_weighted_mat_file = open(edst_weighted_mat_dir, mode='w')
  edst_weighted_trace_file = open(edst_weighted_trace_dir, mode='w')

  edst_lp_topo_file = open(edst_lp_topo_dir, mode='w')
  edst_lp_mat_file = open(edst_lp_mat_dir, mode='w')
  edst_lp_trace_file = open(edst_lp_trace_dir, mode='w')

  sw_to_sw_links = ports - host_per_switch
  switch_objs = []
  for i in range(switches):
    switch_objs.append( Switch(id=switch_ids[i], ports=ports, to_hosts=host_per_switch))

  links = hosts + sw_to_sw_links * switches / 2
  total_nodes = hosts + switches
  # topo_file=open(rrg_topo_dir, mode=w)
  rrg_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )
  rrg_ecmp_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )
  rrg_uniform_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )
  rrg_weighted_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )
  rrg_lp_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )

  edst_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )
  edst_uniform_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )
  edst_weighted_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )
  edst_lp_topo_file.write("%d %d %d %d\n" % (total_nodes, switches, links, ports) )

  rrg_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )
  rrg_ecmp_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )
  rrg_uniform_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )
  rrg_weighted_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )
  rrg_lp_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )

  edst_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )
  edst_uniform_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )
  edst_weighted_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )
  edst_lp_topo_file.write("%s\n" % ( " ".join(map(str, switch_ids)) ) )

  for sw_idx in range(switches):
    for h in range(host_per_switch):
      host_id = sw_idx * host_per_switch + h
      sw_id = switch_ids[sw_idx]
      switch_objs[sw_idx].mark_used(port_id=h)
      rrg_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )
      rrg_ecmp_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )
      rrg_uniform_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )
      rrg_weighted_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )
      rrg_lp_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )

      edst_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )
      edst_uniform_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )
      edst_weighted_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )
      edst_lp_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (host_id, sw_id, 1, 1000, 0, h) )


  link_matrix = np.zeros(shape=(switches, switches), dtype=int)
  ports_conn_matrix = {}
  for i in range(switches):
    link_matrix[i][i] = 1
  
  for i in range(switches):
    ports_conn_matrix[i] = {}
    for j in range(switches):
      ports_conn_matrix[i][j] = []
  
  loop_bg = time.time()
  while switch_has_empty_port(switch_objs) and has_links(link_matrix):
    for src_sw_idx in range(switches):
      dst_idx = random.randint(0, switches - 1)
      ## choose dst switch
      while dst_idx == src_sw_idx:
        dst_idx = random.randint(0, switches - 1)
      if link_matrix[src_sw_idx][dst_idx] == 1 and link_matrix[dst_idx][src_sw_idx] == 1:
        continue
      res, p1, p2 = connect_switches(switch_objs[src_sw_idx], switch_objs[dst_idx])
      # print(res, p1, p2)
      if res == True:
        link_matrix[src_sw_idx][dst_idx] = 1
        link_matrix[dst_idx][src_sw_idx] = 1
        ports_conn_matrix[src_sw_idx][dst_idx].extend([p1, p2])
        ports_conn_matrix[dst_idx][src_sw_idx].extend([p2, p1])

        rrg_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
        rrg_ecmp_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
        rrg_uniform_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
        rrg_weighted_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
        rrg_lp_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
        
        edst_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
        edst_uniform_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
        edst_weighted_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
        edst_lp_topo_file.write("%d %d %.2fGbps %dns 0.000000 %d %d\n" % (switch_ids[src_sw_idx], switch_ids[dst_idx], 1, 1000, p1, p2) )
      
      else:
        continue
    loop_now = time.time()
    if loop_now - loop_bg > 0.5:
      return False
  


  # for sw in switch_objs:
  #   print(sw)
  # ## connect sw and sw
  rrg_topo_file.close()
  rrg_ecmp_topo_file.close()
  rrg_weighted_topo_file.close()
  rrg_uniform_topo_file.close()
  rrg_lp_topo_file.close()

  edst_topo_file.close()
  edst_uniform_topo_file.close()
  edst_weighted_topo_file.close()
  edst_lp_topo_file.close()

  # trace_file = open(trace_dir, mode='w')
  rrg_trace_file.write("%d\n" % (total_nodes))
  rrg_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  rrg_trace_file.close()

  rrg_ecmp_trace_file.write("%d\n" % (total_nodes))
  rrg_ecmp_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  rrg_ecmp_trace_file.close()

  rrg_uniform_trace_file.write("%d\n" % (total_nodes))
  rrg_uniform_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  rrg_uniform_trace_file.close()

  rrg_weighted_trace_file.write("%d\n" % (total_nodes))
  rrg_weighted_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  rrg_weighted_trace_file.close()

  rrg_lp_trace_file.write("%d\n" % (total_nodes))
  rrg_lp_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  rrg_lp_trace_file.close()

  edst_trace_file.write("%d\n" % (total_nodes))
  edst_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  edst_trace_file.close()

  edst_uniform_trace_file.write("%d\n" % (total_nodes))
  edst_uniform_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  edst_uniform_trace_file.close()

  edst_weighted_trace_file.write("%d\n" % (total_nodes))
  edst_weighted_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  edst_weighted_trace_file.close()

  edst_lp_trace_file.write("%d\n" % (total_nodes))
  edst_lp_trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
  edst_lp_trace_file.close()

  ## write mat matrix
  print('write mat')
  """host per switch, switches, switch_to_switch link_num, total_hosts"""
  rrg_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  rrg_uniform_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  rrg_weighted_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  rrg_lp_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  
  edst_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  edst_uniform_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  edst_weighted_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  edst_lp_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  ksp_mat_file.write("%d %d %d %d\n" % (host_per_switch, switches, sw_to_sw_links * switches, hosts) )
  
  """k level clos"""
  print(k_clos)
  rrg_mat_file.write("%d\n" % (k_clos))
  rrg_uniform_mat_file.write("%d\n" % (k_clos))
  rrg_weighted_mat_file.write("%d\n" % (k_clos))
  rrg_lp_mat_file.write("%d\n" % (k_clos))

  edst_mat_file.write("%d\n" % (k_clos))
  edst_uniform_mat_file.write("%d\n" % (k_clos))
  edst_weighted_mat_file.write("%d\n" % (k_clos))
  edst_lp_mat_file.write("%d\n" % (k_clos))

  sw = Switch(id=1, ports=16, to_hosts=4)
  rrg_mat_file.write("%s\n" % (sw.switch_level_port_range()))
  rrg_uniform_mat_file.write("%s\n" % (sw.switch_level_port_range()))
  rrg_weighted_mat_file.write("%s\n" % (sw.switch_level_port_range()))
  rrg_lp_mat_file.write("%s\n" % (sw.switch_level_port_range()))
  
  edst_mat_file.write("%s\n" % (sw.switch_level_port_range()))
  edst_uniform_mat_file.write("%s\n" % (sw.switch_level_port_range()))
  edst_weighted_mat_file.write("%s\n" % (sw.switch_level_port_range()))
  edst_lp_mat_file.write("%s\n" % (sw.switch_level_port_range()))

  for i in range(switches):
    for j in range(switches):
      if i==j or link_matrix[i,j]==0: continue

      rrg_mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
      rrg_uniform_mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
      rrg_weighted_mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
      rrg_lp_mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
      
      edst_mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
      edst_uniform_mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
      edst_weighted_mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
      edst_lp_mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
      
      ksp_mat_file.write("%d %d\n"%(i, j))
  
  ksp_mat_file.close()
  rrg_mat_file.close()
  rrg_uniform_mat_file.close()
  rrg_weighted_mat_file.close()
  rrg_lp_mat_file.close()

  edst_mat_file.close()
  edst_uniform_mat_file.close()
  edst_weighted_mat_file.close()
  edst_lp_mat_file.close()
  
  return True


if __name__== "__main__":
  rrg_mat_fmt = "../experiment/topo/rrg/%d/mat.txt"
  rrg_topo_fmt = "../experiment/topo/rrg/%d/topo.txt"
  rrg_trace_fmt = "../experiment/trace/rrg/%d/trace.txt"

  rrg_ecmp_mat_fmt = "../experiment/topo/rrg-ecmp/%d/mat.txt"
  rrg_ecmp_topo_fmt = "../experiment/topo/rrg-ecmp/%d/topo.txt"
  rrg_ecmp_trace_fmt = "../experiment/trace/rrg-ecmp/%d/trace.txt"

  rrg_uniform_mat_fmt = "../experiment/topo/rrg-uniform/%d/mat.txt"
  rrg_uniform_topo_fmt = "../experiment/topo/rrg-uniform/%d/topo.txt"
  rrg_uniform_trace_fmt = "../experiment/trace/rrg-uniform/%d/trace.txt"

  rrg_weighted_mat_fmt = "../experiment/topo/rrg-weighted/%d/mat.txt"
  rrg_weighted_topo_fmt = "../experiment/topo/rrg-weighted/%d/topo.txt"
  rrg_weighted_trace_fmt = "../experiment/trace/rrg-weighted/%d/trace.txt"

  rrg_lp_mat_fmt = "../experiment/topo/rrg-lp/%d/mat.txt"
  rrg_lp_topo_fmt = "../experiment/topo/rrg-lp/%d/topo.txt"
  rrg_lp_trace_fmt = "../experiment/trace/rrg-lp/%d/trace.txt"

  edst_mat_fmt = "../experiment/topo/edst/%d/mat.txt"
  edst_topo_fmt = "../experiment/topo/edst/%d/topo.txt"
  edst_trace_fmt = "../experiment/trace/edst/%d/trace.txt"

  edst_uniform_mat_fmt = "../experiment/topo/edst-uniform/%d/mat.txt"
  edst_uniform_topo_fmt = "../experiment/topo/edst-uniform/%d/topo.txt"
  edst_uniform_trace_fmt = "../experiment/trace/edst-uniform/%d/trace.txt"

  edst_weighted_mat_fmt = "../experiment/topo/edst-weighted/%d/mat.txt"
  edst_weighted_topo_fmt = "../experiment/topo/edst-weighted/%d/topo.txt"
  edst_weighted_trace_fmt = "../experiment/trace/edst-weighted/%d/trace.txt"

  edst_lp_mat_fmt = "../experiment/topo/edst-lp/%d/mat.txt"
  edst_lp_topo_fmt = "../experiment/topo/edst-lp/%d/topo.txt"
  edst_lp_trace_fmt = "../experiment/trace/edst-lp/%d/trace.txt"


  switches = [64]
  hosts=256
  for sw in switches:
    while True:
      ret = rrg(hosts=hosts, 
                switches=sw, 
                host_per_switch=(hosts/sw), 
                ports=16,
                rrg_topo_dir=rrg_topo_fmt % (sw),
                rrg_mat_dir=rrg_mat_fmt % (sw),
                rrg_trace_dir=rrg_trace_fmt % (sw),

                rrg_ecmp_topo_dir=rrg_ecmp_topo_fmt % (sw),
                rrg_ecmp_trace_dir=rrg_ecmp_trace_fmt % (sw),

                rrg_uniform_topo_dir=rrg_uniform_topo_fmt % (sw),
                rrg_uniform_mat_dir=rrg_uniform_mat_fmt % (sw),
                rrg_uniform_trace_dir=rrg_uniform_trace_fmt % (sw),

                rrg_weighted_topo_dir=rrg_weighted_topo_fmt % (sw),
                rrg_weighted_mat_dir=rrg_weighted_mat_fmt % (sw),
                rrg_weighted_trace_dir=rrg_weighted_trace_fmt % (sw),

                rrg_lp_topo_dir=rrg_lp_topo_fmt % (sw),
                rrg_lp_mat_dir=rrg_lp_mat_fmt % (sw),
                rrg_lp_trace_dir=rrg_lp_trace_fmt % (sw),

                edst_topo_dir=edst_topo_fmt % (sw),
                edst_mat_dir=edst_mat_fmt % (sw),
                edst_trace_dir=edst_trace_fmt % (sw),

                edst_uniform_topo_dir=edst_uniform_topo_fmt % (sw),
                edst_uniform_mat_dir=edst_uniform_mat_fmt % (sw),
                edst_uniform_trace_dir=edst_uniform_trace_fmt % (sw),

                edst_weighted_topo_dir=edst_weighted_topo_fmt % (sw),
                edst_weighted_mat_dir=edst_weighted_mat_fmt % (sw),
                edst_weighted_trace_dir=edst_weighted_trace_fmt % (sw),

                edst_lp_topo_dir=edst_lp_topo_fmt % (sw),
                edst_lp_mat_dir=edst_lp_mat_fmt % (sw),
                edst_lp_trace_dir=edst_lp_trace_fmt % (sw),
              )
      if ret:
        break