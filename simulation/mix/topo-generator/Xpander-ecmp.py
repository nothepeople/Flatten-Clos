#coding=utf-8
import sys
import numpy as np
from numpy import linalg as LA

def get_lambda2(mat):
    eig,vecs = LA.eig(mat)
    eig = np.abs(eig)
    eig.sort()
    return eig[-2]

def get_spectral_gap(d):
    return 2*np.sqrt(d-1)

def is_ramanujan(mat,d):
    return get_lambda2(mat) < get_spectral_gap(d)

# d= the degree of the graph
# k= number of lifts to perform
# e.g.,: random_k_lift(4,6) will create a 4 regualr graph with 30 nodes
def random_k_lift(d, k):
  num_nodes = (d+1)*k
  mat = np.zeros( (num_nodes,num_nodes) )
  # print(num_nodes, mat)
  # go over all meta nodes
  for meta1 in range(d+1):
    # connect to any other meta node
    for meta2 in range(meta1+1, d+1):
      # connect the ToRs between the meta-nodes randomally
      perm = np.random.permutation(k)
      for src_ind in range(k):
        src = meta1*k + src_ind
        dst = meta2*k + perm[src_ind]

        # connect the link
        mat[src,dst] = 1
        mat[dst,src] = 1
  # print(mat)
  if not is_ramanujan(mat,d):
      # try again if we got a bad Xpander
      return random_k_lift(d,k)
  return mat

"""
dlin
"""
def Xpander_topo_with_host(switch_num, d, hosts_per_switch=1, total_hosts=256, host_bw=1, sw2sw_bw=1, topo_dir="topo.txt", trace_dir="", mat_dir="", ):
  # topo connection info
  if switch_num%(d+1) != 0:
        print("This script supports only multiplications of d+1 (the degree plus 1), now quitting")
        sys.exit(0)
  mat = random_k_lift(d, switch_num//(d+1))
  # np.savetxt(outname, mat, delimiter=delim)
  with open(mat_dir, 'w') as f:
    """host per switch, total_hosts, switch_to_switch link_num"""
    f.write("%d %d %d %d\n" % (hosts_per_switch, switch_num, switch_num * d, total_hosts) )
    for i in range(switch_num):
        for j in range(switch_num):
            if i==j or mat[i,j]==0: continue
            f.write("%d %d\n"%(i,j))

  host_num = total_hosts
  total_nodes = host_num + switch_num
  host_ids = [id for id in range(host_num)]
  switch_ids = [id + host_num for id in range(switch_num)]

  """ write trace nodes """
  node_ids = []
  node_ids.extend(host_ids)
  node_ids.extend(switch_ids)
  with open(trace_dir, 'w') as f:
    f.write("%d\n" % len(node_ids))
    f.write("%s\n" % (" ".join(map(str, node_ids))) )

  link_num = switch_num * d / 2 + total_hosts
  topo = open(topo_dir, mode='w')
  topo.write("%d %d %d\n" % (total_nodes, switch_num, link_num) )
  topo.write("%s\n" % (" ".join(map(str, switch_ids))))

  """step1: connect host with switch"""
  for host_id in host_ids:
    topo.write("%d %d %.2fGbps %dns 0.000000\n" % (host_id, switch_ids[host_id // hosts_per_switch ], host_bw, 1000))
  
  """step 2: connect switch with switch"""
  for i in range(switch_num):
    for j in range(switch_num):
      if i > j:
        if mat[i, j] == 1:
          topo.write("%d %d %.2fGbps %dns 0.000000\n" % (i + host_num, j + host_num, sw2sw_bw, 1000))
  topo.close()

if __name__ == "__main__":
  # used for k-shortest path
  mat_fmt = "../experiment/topo/Xpander-ecmp/%s/mat.txt"
  topo_fmt = "../experiment/topo/Xpander-ecmp/%s/topo.txt"
  trace_fmt = "../experiment/trace/Xpander-ecmp/%s/trace.txt"
  MNs = [4]
  ratios = [1.0]
  bws = ["1G"]
  up_down_ratios = ["7:4", "7:6", "7:8"]
  Xpander_topo_with_host(32, 
              d=7,
              hosts_per_switch=8, 
              total_hosts=256, 
              host_bw=1,
              sw2sw_bw=1.14,
              
              topo_dir=topo_fmt % ("7:8"),
              trace_dir=trace_fmt % ("7:8"),
              mat_dir=mat_fmt % ("7:8"))
  Xpander_topo_with_host(48, 
              d=7, 
              hosts_per_switch=6,
              total_hosts=256, 
              host_bw=1,
              sw2sw_bw=1.42,
              topo_dir=topo_fmt % ("7:6"),
              trace_dir=trace_fmt % ("7:6"),
              mat_dir=mat_fmt % ("7:6"))
  Xpander_topo_with_host(64, 
              d=7, 
              hosts_per_switch=4,
              total_hosts=256, 
              host_bw=1,
              sw2sw_bw=1.71,
              topo_dir=topo_fmt % ("7:4"),
              trace_dir=trace_fmt % ("7:4"),
              mat_dir=mat_fmt % ("7:4"))