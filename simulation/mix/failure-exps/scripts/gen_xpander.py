import sys
import numpy as np
from numpy import linalg as LA
import random

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

    if not is_ramanujan(mat,d):
        # try again if we got a bad Xpander
        return random_k_lift(d,k)

    return mat

def get_sw_unoccupied_port(sw, to_hosts, port_occupy_map):
    ports = port_occupy_map[sw][to_hosts:]
    rnd = random.randint(0, len(ports)-1)
    selected_port_no = to_hosts +rnd
    while port_occupy_map[sw][selected_port_no]:
      rnd = random.randint(0, len(ports)-1)
      selected_port_no = to_hosts +rnd
    return selected_port_no

def get_conn_port(switches, topo_matrix, to_hosts=4, ports=16):
    port_conn_matrix = {}
    for sA in range(switches):
        port_conn_matrix[sA] = {}
        for sB in range(switches):
            port_conn_matrix[sA][sB] = []
    
    port_occupy_map = {}
    for sid in range(switches):
        port_occupy_map[sid] = [False for _ in range(ports)]

    for sA in range(switches):
        for sB in range(sA + 1, switches):
            if topo_matrix[sA][sB] == 1:
                pA = get_sw_unoccupied_port(sA, to_hosts, port_occupy_map)
                pB = get_sw_unoccupied_port(sB, to_hosts, port_occupy_map)
                port_occupy_map[sA][pA] = True
                port_occupy_map[sB][pB] = True 
                port_conn_matrix[sA][sB].extend([pA, pB])
                port_conn_matrix[sB][sA].extend([pB, pA])
    return port_conn_matrix


def Xpander_topo_gen(switches, ports, to_hosts, mat_dir=None):
    port_degree = ports - to_hosts
    if switches % (port_degree + 1) != 0:
        print("This script supports only multiplications of d+1 (the degree plus 1), now quitting")
        sys.exit(0)
    topo_matrix = random_k_lift(port_degree, switches // (port_degree + 1))
    ports_conn_matrix = get_conn_port(switches, topo_matrix, to_hosts=0, ports=port_degree)
    if mat_dir is not None:
        print("Write Xpander mat")
        mat_file = open(mat_dir, mode='w')
        mat_file.write("%d %d %d %d\n" % (to_hosts, switches, (ports - to_hosts) * switches, switches * to_hosts) )
        for i in range(switches):
            for j in range(switches):
                if i==j or topo_matrix[i,j]==0: continue
            # print(ports_conn_matrix[i][j])
                mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
        mat_file.close()
    return topo_matrix, ports_conn_matrix


def main(outname, d, n, delim=','):
    np.savetxt(outname, mat, delimiter=delim)
    with open(outname+"_mat", 'w') as f:
        for i in range(n):
            for j in range(n):
                if i==j or mat[i,j]==0: continue
                f.write("%d %d\n"%(i,j))

if __name__ == "__main__":
    topo_matrix, port_conn_matrix = Xpander_topo_gen(6, 2)
    print(topo_matrix)
    print(port_conn_matrix)
    # args = sys.argv[1:]
    # if len(args) != 3:
    #     print("Usage: gen_xpander.py <out_file> <switch network degree>(int) <num switches>(int)")
    # else:
    #     main(args[0], int(args[1]), int(args[2]))