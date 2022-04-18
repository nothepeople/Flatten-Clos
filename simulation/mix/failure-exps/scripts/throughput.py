#!/usr/bin/env python3
from math import ceil
from gurobipy import *
import numpy as np
from fc import fc_topo_gen, convert_into_vclos, paths_of_any_pairs, do_topo_gen
from ecmp import cal_shortest_path
import os
import random
from multiprocessing import Process
from deadlock_detection import deadlock_detection
import networkx as nx
import copy

import sys
sys.path.append('TUB')
from topo_repo import topology

class Edge:
    def __init__(self, nodeA, nodeB):
        self.nodeA = nodeA
        self.nodeB = nodeB

    def __hash__(self):
        return hash(self.nodeA) + hash(self.nodeB)

    def __eq__(self, other):
        return  type(self) == type(other) and self.nodeA == other.nodeA and self.nodeB == other.nodeB 
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f'({self.nodeA},{self.nodeB})'

def Throughput(traffic_matrix, path_pair, label="a2a"):
    switches = len(traffic_matrix)
    edges = []

    for src in range(switches):
        for dst in range(switches):
            for p in path_pair[ src ][ dst ]:
                for idx in range(len(p) - 1):
                    edges.append( Edge(nodeA=p[idx], nodeB=p[idx + 1]) )

    ### deduplicate
    edges = set(edges)

    # print(f'edges: {len(edges)}')
    m = Model("throughput-model")
    # solver = pywraplp.Solver.CreateSolver('GLOP')
    
    throughput = m.addVar(lb=0.0, ub=float('inf'), name='throughput')
    
    fvars = {}
    for src in range(switches):
        fvars[ src ] = {}
        for dst in range(switches):
            fvars [ src ][ dst ] = []
            for p in range( len(path_pair[src][dst]) ):
                fvars [ src ][ dst ].append( m.addVar(lb=0.0, ub=float('inf'), name=f'{src}_{dst}_{p}') )
                # fvars [ src ][ dst ].append( solver.NumVar(1.0 / len(path_pair[src][dst]), 1.0 / len(path_pair[src][dst]), f'{src}_{dst}_{p}') )
    # print('Number of variables =', solver.NumVariables())
    ### Add constraint
    for src in range(switches):
        for dst in range(switches):
            if src == dst: continue
            constraint = 0
            for p in range( len(path_pair[src][dst]) ):
                constraint += fvars[ src ][ dst ][ p ]
            m.addConstr(constraint - throughput * traffic_matrix[ src ][ dst ] == 0)

    ### Edge capacity constraint
    # print("edge capacity constr")
    for e in edges:
        constraint = 0
        for src in range(switches):
            for dst in range(switches):
                if src == dst: continue
                for idx in range( len(path_pair[src][dst]) ):
                    path = path_pair[ src ][ dst ][ idx ]
                    
                    has_edge = False
                    for pidx in  range( len(path) - 1):
                        if e == Edge(nodeA=path[pidx], nodeB=path[pidx + 1]):
                            has_edge = True
                            break
                    if has_edge:
                        
                        constraint += fvars[ src ][ dst ][ idx ]
        m.addConstr(constraint <= 1)
    # print('Number of constraints =', solver.NumConstraints())
    
    m.setObjective(throughput * 1 , GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()
    if  m.status == GRB.OPTIMAL:
        # print('Solution:')
        # print('hi')
        print('%s Throughput value = %.6f' % (label, m.ObjVal ) )
    else:
        print('The problem does not have an optimal solution.')
    return throughput

def gen_all_to_all_TM(switches, switch_recv_bandwidth):
    tm = np.zeros(shape=(switches, switches))
    for i in range(switches):
        for j in range(switches):
            if i == j: continue
            # tm[i][j] = 1
            tm[i][j] = switch_recv_bandwidth / float(switches - 1)
    # print(tm)
    return tm

def gen_one_to_one_TM(switches, switch_recv_bandwidth):
    tm = np.zeros(shape=(switches, switches))
    for i in range(switches):
        j = random.randint(0, switches - 1)
        while i == j or tm[i][j] > 0 or np.sum(tm[:, j]) > 0:
            j = random.randint(0, switches - 1)
        tm[i][j] = switch_recv_bandwidth
    # print(tm)
    return tm

def gen_skewed_TM(switches, switch_recv_bandwidth, theta, phi):
    tm = np.zeros(shape=(switches, switches))
    ### theta: fraction of hot racks
    ### phi: concentrated traffic at hot rack switches
    N_hot = int(ceil(switches * theta))
    N_cold = switches - N_hot
    # print(f"hot racks : {N_hot}")
    hot_rack_ids = []
    while len(hot_rack_ids) < N_hot:
        rack_id = random.randint(0, switches - 1)
        while rack_id in hot_rack_ids:
            rack_id = random.randint(0, switches - 1)
        hot_rack_ids.append(rack_id)
    # print("hot rack ids: ", hot_rack_ids)

    p_hot_to_hot = phi * phi / (N_hot * N_hot)
    p_cold_to_cold = (1 - phi) * (1 - phi) / (N_cold * N_cold)
    p_hot_to_cold = phi * (1 - phi) / (N_cold * N_hot)
    # print(f"p_hot_to_hot: {p_hot_to_hot}, p_cold_to_cold: {p_cold_to_cold}, p_hot_to_cold: {p_hot_to_cold}")
    s_hot = 0
    for i in range(switches):
        for j in range(switches):
            if i == j: continue
            if i in hot_rack_ids and j in hot_rack_ids:
                tm[i][j] = p_hot_to_hot * switch_recv_bandwidth * switches * 0.6
                s_hot += tm[i][j]
            elif not i in hot_rack_ids and not j in hot_rack_ids:
                tm[i][j] = p_cold_to_cold * switch_recv_bandwidth * switches * 0.6
            else:
                tm[i][j] = p_hot_to_cold * switch_recv_bandwidth * switches * 0.6
                s_hot += tm[i][j]
    # print(f"hot / sum: {s_hot / np.sum(tm)}")
    return tm

def test_throughput_calculation():
    e1 = Edge(1,0)
    e2 = Edge(1,0)
    print(e1 == e2)
    traffic_matrix = [ 
        [0, 0.45, 0.175],
        [0.3, 0, 0.35],
        [0.65, 0.3, 0]
    ]

    path_pair = {}
    for i in range(3):
        path_pair[i] = {}
        for j in range(3):
            path_pair[i][j] = []
    
    path_pair[0][1].append( [0,2,1] )
    path_pair[0][2].append( [0,1,2] )
    path_pair[1][0].append( [1,2,0] )
    path_pair[1][2].append( [1,0,2] )
    path_pair[2][0].append( [2,1,0] )
    path_pair[2][1].append( [2,0,1] )

    Throughput(traffic_matrix, path_pair)

def read_edst_path(path_file='path'):
    path_f = open(path_file, mode='r')
    lines = path_f.read().splitlines()
    path_f.close()
            
    idx = 0
    l0 = [int(x) for x in lines[idx].split(' ')]
    pairs = l0[0]
    host_per_sw = l0[1]
    switches = l0[2]
    print("sw", switches)

    edst_path_pair = {}
    for i in range(switches):
        edst_path_pair[i] = {}
        for j in range(switches):
            edst_path_pair[i][j] = []
            

    idx += 1
    for _ in range(pairs):
        l = [int(x) for x in lines[idx].split(' ')]
        idx +=1
        src = l[0]
        dst = l[1]
        path_num = l[2]
        for _ in range(path_num):
            edst_path_pair[ src ][ dst ].append([int(x) for x in lines[idx].split(' ')][ 1 : ])
            idx += 1
    return edst_path_pair



def all_to_all_throughput():
    print("start a2a process")
    random.seed(12)
    switches = 288
    ports = 24
    to_hosts = 4
    ports_of_vir_layer=[3, 7, 7, 3]

    topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
    ## virtual up-down path
    vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
    virtual_up_down_path = paths_of_any_pairs(vclos, len(ports_of_vir_layer), switches)
    
    print("Calculate Edge-Disjoint-Spanning-Trees...")
    # os.system("g++ edst.cc -w -std=c++11 -o edst")
    # os.system("./edst")
    edst_path_pair = read_edst_path()

    print("Calculating shortest paths ...")
    ## calculate shortest path
    shortest_paths_of_any_pair = cal_shortest_path(topo_matrix, switches)

    # print("Calculating Clos shortest path ...")
    # clos_matrix, sw = clos_mat()
    # clos_ecmp_path = cal_shortest_path(clos_matrix, sw)

    print("TEST ALL TO ALL TRAFFIC STARTING...\n")

    to_hosts = 1
    print(f"to hosts = {to_hosts}")
    tm = gen_all_to_all_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="a2a + ECMP")
    Throughput(tm, virtual_up_down_path, label="a2a + UP-DOWN")
    Throughput(tm, edst_path_pair, label="a2a + EDST")
    # Throughput(tm, clos_ecmp_path, label="a2a + CLOS")

    to_hosts = 4
    print(f"to hosts = {to_hosts}")
    tm = gen_all_to_all_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="a2a + ECMP")
    Throughput(tm, virtual_up_down_path, label="a2a + UP-DOWN")
    Throughput(tm, edst_path_pair, label="a2a + EDST")
    # Throughput(tm, clos_ecmp_path, label="a2a + CLOS")

    
    to_hosts = 8
    print(f"to hosts = {to_hosts}")
    tm = gen_all_to_all_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="a2a + ECMP")
    Throughput(tm, virtual_up_down_path, label="a2a + UP-DOWN")
    Throughput(tm, edst_path_pair, label="a2a + EDST")
    # Throughput(tm, clos_ecmp_path, label="a2a + CLOS")

    to_hosts = 12
    print(f"to hosts = {to_hosts}")
    tm = gen_all_to_all_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="a2a + ECMP")
    Throughput(tm, virtual_up_down_path, label="a2a + UP-DOWN")
    Throughput(tm, edst_path_pair, label="a2a + EDST")
    # Throughput(tm, clos_ecmp_path, label="a2a + CLOS")

    to_hosts = 16
    print(f"to hosts = {to_hosts}")
    tm = gen_all_to_all_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="a2a + ECMP")
    Throughput(tm, virtual_up_down_path, label="a2a + UP-DOWN")
    Throughput(tm, edst_path_pair, label="a2a + EDST")
    # Throughput(tm, clos_ecmp_path, label="a2a + CLOS")

    to_hosts = 18
    print(f"to hosts = {to_hosts}")
    tm = gen_all_to_all_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="a2a + ECMP")
    Throughput(tm, virtual_up_down_path, label="a2a + UP-DOWN")
    Throughput(tm, edst_path_pair, label="a2a + EDST")
    # Throughput(tm, clos_ecmp_path, label="a2a + CLOS")

   
def one_to_one_throughput():
    print("start o2o process")
    random.seed(120)
    switches = 192
    ports = 24
    to_hosts = 6
    ports_of_vir_layer=[3, 6, 6, 3]

    topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
    ## virtual up-down path
    vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
    virtual_up_down_path = paths_of_any_pairs(vclos, len(ports_of_vir_layer), switches)
    
    # print("Calculate Edge-Disjoint-Spanning-Trees...")
    # os.system("g++ edst.cc -w -std=c++11 -o edst")
    # os.system("./edst")
    edst_path_pair = read_edst_path()
    # print(edst_path_pair[ 0 ][ 1 ])

    print("Calculating shortest paths ...")
    ## calculate shortest path
    shortest_paths_of_any_pair = cal_shortest_path(topo_matrix, switches)
    
    # print("Calculating Clos shortest path ...")
    # clos_mat, switches = clos_mat()
    # clos_ecmp_path = cal_shortest_path(clos_mat, switches)

    print("TEST STARTING...\n")


    ###
    print("TEST ONE TO ONE TRAFFIC STARTING...")
    to_hosts = 1
    print(f"to hosts = {to_hosts}")
    tm = gen_one_to_one_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="o2o + ECMP")
    Throughput(tm, virtual_up_down_path, label="o2o + UP-DOWN")
    Throughput(tm, edst_path_pair, label="o2o + EDST")
    # Throughput(tm, clos_ecmp_path, label="o2o + CLOS")
    
    to_hosts = 4
    print(f"to hosts = {to_hosts}")
    tm = gen_one_to_one_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="o2o + ECMP")
    Throughput(tm, virtual_up_down_path, label="o2o + UP-DOWN")
    Throughput(tm, edst_path_pair, label="o2o + EDST")
    # Throughput(tm, clos_ecmp_path, label="o2o + CLOS")

    
    to_hosts = 8
    print(f"to hosts = {to_hosts}")
    tm = gen_one_to_one_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="o2o + ECMP")
    Throughput(tm, virtual_up_down_path, label="o2o + UP-DOWN")
    Throughput(tm, edst_path_pair, label="o2o + EDST")
    # Throughput(tm, clos_ecmp_path, label="o2o + CLOS")

    to_hosts = 12
    print(f"to hosts = {to_hosts}")
    tm = gen_one_to_one_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="o2o + ECMP")
    Throughput(tm, virtual_up_down_path, label="o2o + UP-DOWN")
    Throughput(tm, edst_path_pair, label="o2o + EDST")
    # Throughput(tm, clos_ecmp_path, label="o2o + CLOS")

    to_hosts = 16
    print(f"to hosts = {to_hosts}")
    tm = gen_one_to_one_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="o2o + ECMP")
    Throughput(tm, virtual_up_down_path, label="o2o + UP-DOWN")
    Throughput(tm, edst_path_pair, label="o2o + EDST")
    # Throughput(tm, clos_ecmp_path, label="o2o + CLOS")

    to_hosts = 18
    print(f"to hosts = {to_hosts}")
    tm = gen_one_to_one_TM(switches, to_hosts)
    Throughput(tm, shortest_paths_of_any_pair, label="o2o + ECMP")
    Throughput(tm, virtual_up_down_path, label="o2o + UP-DOWN")
    Throughput(tm, edst_path_pair, label="o2o + EDST")
    # Throughput(tm, clos_ecmp_path, label="o2o + CLOS")

    print("TEST ONE TO ONE TRAFFIC ENDING...")


def skewed_throughput():
    print("Start skewed process")
    random.seed(120)
    switches = 192
    ports = 24
    to_hosts = 6
    ports_of_vir_layer=[3, 6, 6, 3]

    topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
    ## virtual up-down path
    vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
    virtual_up_down_path = paths_of_any_pairs(vclos, len(ports_of_vir_layer), switches)
    
    # print("Calculate Edge-Disjoint-Spanning-Trees...")
    # os.system("g++ edst.cc -w -std=c++11 -o edst")
    # os.system("./edst")
    edst_path_pair = read_edst_path()

    print("Calculating shortest paths ...")
    ## calculate shortest path
    shortest_paths_of_any_pair = cal_shortest_path(topo_matrix, switches)

    # print("Calculating Clos shortest path ...")
    # clos_matrix, switches = clos_mat()
    # clos_ecmp_path = cal_shortest_path(clos_matrix, switches)

    print("TEST SKEWED TRAFFIC STARTING...\n")

    to_hosts = 1
    print(f"to hosts = {to_hosts}")
    tm = gen_skewed_TM(switches, to_hosts, theta=0.04, phi=0.75)
    Throughput(tm, shortest_paths_of_any_pair, label="skewed + ECMP")
    Throughput(tm, virtual_up_down_path, label="skewed + UP-DOWN")
    Throughput(tm, edst_path_pair, label="skewed + EDST")
    # Throughput(tm, clos_ecmp_path, label="skewed + CLOS")

    to_hosts = 4
    print(f"to hosts = {to_hosts}")
    tm = gen_skewed_TM(switches, to_hosts, theta=0.04, phi=0.75)
    Throughput(tm, shortest_paths_of_any_pair, label="skewed + ECMP")
    Throughput(tm, virtual_up_down_path, label="skewed + UP-DOWN")
    Throughput(tm, edst_path_pair, label="skewed + EDST")
    # Throughput(tm, clos_ecmp_path, label="skewed + CLOS")

    
    to_hosts = 8
    print(f"to hosts = {to_hosts}")
    tm = gen_skewed_TM(switches, to_hosts, theta=0.04, phi=0.75)
    Throughput(tm, shortest_paths_of_any_pair, label="skewed + ECMP")
    Throughput(tm, virtual_up_down_path, label="skewed + UP-DOWN")
    Throughput(tm, edst_path_pair, label="skewed + EDST")
    # Throughput(tm, clos_ecmp_path, label="skewed + CLOS")

    to_hosts = 12
    print(f"to hosts = {to_hosts}")
    tm = gen_skewed_TM(switches, to_hosts, theta=0.04, phi=0.75)
    Throughput(tm, shortest_paths_of_any_pair, label="skewed + ECMP")
    Throughput(tm, virtual_up_down_path, label="skewed + UP-DOWN")
    Throughput(tm, edst_path_pair, label="skewed + EDST")
    # Throughput(tm, clos_ecmp_path, label="skewed + CLOS")


    to_hosts = 16
    print(f"to hosts = {to_hosts}")
    tm = gen_skewed_TM(switches, to_hosts, theta=0.04, phi=0.75)
    Throughput(tm, shortest_paths_of_any_pair, label="skewed + ECMP")
    Throughput(tm, virtual_up_down_path, label="skewed + UP-DOWN")
    Throughput(tm, edst_path_pair, label="skewed + EDST")
    # Throughput(tm, clos_ecmp_path, label="skewed + CLOS")

    to_hosts = 18
    print(f"to hosts = {to_hosts}")
    tm = gen_skewed_TM(switches, to_hosts, theta=0.04, phi=0.75)
    Throughput(tm, shortest_paths_of_any_pair, label="skewed + ECMP")
    Throughput(tm, virtual_up_down_path, label="skewed + UP-DOWN")
    Throughput(tm, edst_path_pair, label="skewed + EDST")
    # Throughput(tm, clos_ecmp_path, label="skewed + CLOS")


def worst_throughput_task(process_label, worst_case_tm, to_host, virtual_up_down_path, edst_path_pair, shortest_paths_of_any_pair):
    # print("started", process_label)
    Throughput(worst_case_tm, shortest_paths_of_any_pair, label="{} + ECMP".format(process_label))
    Throughput(worst_case_tm, virtual_up_down_path, label="{} + UP-DOWN".format(process_label))
    Throughput(worst_case_tm, edst_path_pair, label="{} + EDST".format(process_label))

def maximal_permutation_matrix_throughput():
    host_per_switch_list = [1, 4, 8, 12, 16, 20]

    random.seed(7)
    ports_of_vir_layer = [3,7,7,3]
    layers=len(ports_of_vir_layer)
    switches=288
    ports=24
    to_hosts=4
    
    topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)

    vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
    virtual_up_down_path = paths_of_any_pairs(vclos, len(ports_of_vir_layer), switches)
    
    print("Calculate Edge-Disjoint-Spanning-Trees...")
    os.system("g++ edst.cc -w -std=c++11 -o edst")
    os.system("./edst")
    edst_path_pair = read_edst_path()

    print("Calculating shortest paths ...")
    ## calculate shortest path
    shortest_paths_of_any_pair = cal_shortest_path(topo_matrix, switches)

    G = nx.Graph()
    for src in range(switches):
        for dst in range(switches):
            if topo_matrix[src][dst] == 1:
                G.add_edge(src, dst)

    print("G's nodes: ", G.number_of_nodes())
    print("G's edges: ", G.number_of_edges())
    
    tor_list = [tor_sid for tor_sid in range(switches)]
    
    process_list = []
    for hosts in host_per_switch_list:
        demand_list = {}

        for sid in range(switches):
            demand_list[sid] = hosts

        topo = topology.Topology(G, tor_list, demand_list)
        tub = topo.get_tub()
        print("hosts = {}, TUB = {}".format(hosts, tub))

        worst_case_tm = np.zeros((switches, switches))
        
        ## tm is a dictionary
        tm, _ = topo.get_near_worst_case_traffic_matrix()

        for key in tm.keys():
            src = key[0]
            dst = key[1]
            # print(tm[key])
            worst_case_tm[src][dst] = tm[key]
        
        copied_virtual_up_down_path = copy.deepcopy(virtual_up_down_path)
        copied_edst_path_pair = copy.deepcopy(edst_path_pair)
        copied_shortest_path_pair = copy.deepcopy(shortest_paths_of_any_pair)
        p = Process(target=worst_throughput_task, args=("P_host_{}".format(hosts), 
                worst_case_tm, hosts, 
                copied_virtual_up_down_path, 
                copied_edst_path_pair, 
                copied_shortest_path_pair))
        process_list.append(p)
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()

def test_throughput():
    a2a_process = Process(target=all_to_all_throughput)
    o2o_process = Process(target=one_to_one_throughput)
    skewed_process = Process(target=skewed_throughput)

    a2a_process.start()
    o2o_process.start()
    skewed_process.start()

    a2a_process.join()
    o2o_process.join()
    skewed_process.join()

def test_deadlock():
    random.seed(12)
    switches = 192
    ports = 24
    to_hosts = 6
    ports_of_vir_layer=[3, 6, 6, 3]

    topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
    ## virtual up-down path
    vclos = convert_into_vclos(switch_objs, topo_matrix, ports_conn_matrix, ports_of_vir_layer)
    
    print("Calculate Edge-Disjoint-Spanning-Trees...")
    os.system("g++ edst.cc -w -std=c++11 -o edst")
    os.system("./edst")
    edst_path_pair = read_edst_path()
    path_list = []
    for i in range(switches):
        for j in range(switches):
            for p in edst_path_pair[i][j]:
                path_list.append(p)
    has_cycle ,_,_,_ = deadlock_detection(path_list, switches, ports, ports_conn_matrix)
    if has_cycle:
        print("deadlock")
    else:
        print("no")


if __name__ == "__main__":
    # test_throughput()
    maximal_permutation_matrix_throughput()
