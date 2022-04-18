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
from gen_jellyfish import jellyfish_topo_gen
from gen_xpander import Xpander_topo_gen

import sys
sys.path.append('TUB')
from topo_repo import topology
from throughput import Throughput


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


def get_worst_tm_and_tub(topo_type, switches, topo_matrix, hosts):
    G = nx.Graph()
    for src in range(switches):
        for dst in range(switches):
            if topo_matrix[src][dst] == 1:
                G.add_edge(src, dst)
    
    tor_list = [tor_sid for tor_sid in range(switches)]
    demand_list = {}
    for sid in range(switches):
        demand_list[sid] = hosts

    topo = topology.Topology(G, tor_list, demand_list)
    tub = topo.get_tub()
    print("topo_type=%s hosts=%s, TUB=%.2f" % (topo_type, hosts, tub))

    worst_case_tm = np.zeros((switches, switches))
    
    ## tm is a dictionary
    tm, _ = topo.get_near_worst_case_traffic_matrix()

    for key in tm.keys():
        src = key[0]
        dst = key[1]
        # print(tm[key])
        worst_case_tm[src][dst] = tm[key]
    return worst_case_tm, tub

def gen_all_to_all_TM(switches, switch_recv_bandwidth):
    tm = np.zeros(shape=(switches, switches))
    for i in range(switches):
        for j in range(switches):
            if i == j: continue
            # tm[i][j] = 1
            tm[i][j] = switch_recv_bandwidth / float(switches - 1)
    # print(tm)
    return tm

def worst_throughput_task(process_label, worst_case_tm, to_host, virtual_up_down_path, edst_path_pair, shortest_paths_of_any_pair):
    # print("started", process_label)
    Throughput(worst_case_tm, shortest_paths_of_any_pair, label="{} + ECMP".format(process_label))
    Throughput(worst_case_tm, virtual_up_down_path, label="{} + UP-DOWN".format(process_label))
    Throughput(worst_case_tm, edst_path_pair, label="{} + EDST".format(process_label))

def path_calculation(type, mat_dir, path_dir):
    if type == "ksp": os.system("./ksp %s %s" % (mat_dir, path_dir))
    else: os.system("./edst %s %s" % (mat_dir, path_dir))

def gen_all_to_all_TM(switches, switch_recv_bandwidth):
    tm = np.zeros(shape=(switches, switches))
    for i in range(switches):
        for j in range(switches):
            if i == j: continue
            # tm[i][j] = 1
            tm[i][j] = switch_recv_bandwidth / float(switches - 1)
    # print(tm)
    return tm

def worst_throughput():
    host_per_switch_list = [6, 8, 12, 18]
    random.seed(1)
    ports_of_vir_layer = [3,6,6,3]
    layers=len(ports_of_vir_layer)
    switches=285
    ports=24
    to_hosts=6
    
    """
    host_per_switch_list = [4]
    # xrandom.seed(1)
    ports_of_vir_layer = [2,4,4,2]
    layers=len(ports_of_vir_layer)
    switches=65
    ports=16
    to_hosts=4
    """

    fc_mat_dir = "throughput/worst/fc/mat.txt"
    jf_mat_dir = "throughput/worst/jf/mat.txt"
    xpander_mat_dir = "throughput/worst/xpander/mat.txt"

    fc_ksp_path_dir = "throughput/worst/fc/ksp_path.txt"
    jf_ksp_path_dir = "throughput/worst/jf/ksp_path.txt"
    xpander_ksp_path_dir =  "throughput/worst/xpander/ksp_path.txt"

    fc_edst_path_dir = "throughput/worst/fc/edst_path.txt"
    jf_edst_path_dir = "throughput/worst/jf/edst_path.txt"
    xpander_edst_path_dir =  "throughput/worst/xpander/edst_path.txt"

    jf_topo_matrix, _ = jellyfish_topo_gen(switches, ports, to_hosts, jf_mat_dir)
    xpander_topo_matrix, _ = Xpander_topo_gen(switches, ports, to_hosts, xpander_mat_dir)

    fc_topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer, fc_mat_dir)
    print("FC UP-DOWN PATH CALCULATION...")
    vclos = convert_into_vclos(switch_objs, fc_topo_matrix, ports_conn_matrix, ports_of_vir_layer)
    fc_up_down_path = paths_of_any_pairs(vclos, len(ports_of_vir_layer), switches)
    
    fc_ecmp_path = cal_shortest_path(fc_topo_matrix, switches)
    jf_ecmp_path = cal_shortest_path(jf_topo_matrix, switches)
    xpander_ecmp_path = cal_shortest_path(xpander_topo_matrix, switches) 

    print("CALCULATE EDST AND KSP PATHS...")
    os.system("g++ edst.cc -w -std=c++11 -o edst")
    os.system("g++ k-short-path.cc -w -std=c++11 -lpthread -o ksp")

    path_processes = []
    path_processes.append(Process(target=path_calculation, args=("ksp", fc_mat_dir, fc_ksp_path_dir)))
    path_processes.append(Process(target=path_calculation, args=("ksp", jf_mat_dir, jf_ksp_path_dir)))
    path_processes.append(Process(target=path_calculation, args=("ksp", xpander_mat_dir, xpander_ksp_path_dir)))
    
    path_processes.append(Process(target=path_calculation, args=("edst", fc_mat_dir, fc_edst_path_dir)))
    path_processes.append(Process(target=path_calculation, args=("edst", jf_mat_dir, jf_edst_path_dir)))
    path_processes.append(Process(target=path_calculation, args=("edst", xpander_mat_dir, xpander_edst_path_dir)))
    
    for p in path_processes:
        p.start()

    for p in path_processes:
        p.join()

    print("READ EDST AND KSP PATHS...")
    fc_edst_path = read_edst_path(fc_edst_path_dir)
    jf_edst_path = read_edst_path(jf_edst_path_dir)
    xpander_edst_path = read_edst_path(xpander_edst_path_dir)

    fc_ksp_path = read_edst_path(fc_ksp_path_dir)
    jf_ksp_path = read_edst_path(jf_ksp_path_dir)
    xpander_ksp_path = read_edst_path(xpander_ksp_path_dir)

    worst_processes = []
    for host in host_per_switch_list:
        fc_worst_case_tm, _ = get_worst_tm_and_tub("FC", switches, fc_topo_matrix, host)
        jf_worst_case_tm, _ = get_worst_tm_and_tub("JF", switches, jf_topo_matrix, host)
        xpander_worst_case_tm, _ = get_worst_tm_and_tub("XP", switches, xpander_topo_matrix, host)
        worst_processes.append(Process(target=Throughput, args=(fc_worst_case_tm, fc_up_down_path, "WORST %d FC UP_DOWN" % (host) )))
        
        worst_processes.append(Process(target=Throughput, args=(fc_worst_case_tm, fc_ecmp_path, "WORST %d FC ECMP" % (host) )))
        worst_processes.append(Process(target=Throughput, args=(fc_worst_case_tm, fc_edst_path, "WORST %d FC EDST" % (host) )))
        worst_processes.append(Process(target=Throughput, args=(fc_worst_case_tm, fc_ksp_path, "WORST %d FC KSP" % (host) )))
        
        worst_processes.append(Process(target=Throughput, args=(jf_worst_case_tm, jf_ecmp_path, "WORST %d JF ECMP" % (host) )))
        worst_processes.append(Process(target=Throughput, args=(jf_worst_case_tm, jf_edst_path, "WORST %d JF EDST" % (host) )))
        worst_processes.append(Process(target=Throughput, args=(jf_worst_case_tm, jf_ksp_path, "WORST %d JF KSP" % (host) )))
        
        worst_processes.append(Process(target=Throughput, args=(xpander_worst_case_tm, xpander_ecmp_path, "WORST %d XP ECMP" % (host) )))
        worst_processes.append(Process(target=Throughput, args=(xpander_worst_case_tm, xpander_edst_path, "WORST %d XP EDST" % (host) )))
        worst_processes.append(Process(target=Throughput, args=(xpander_worst_case_tm, xpander_ksp_path, "WORST %d XP KSP" % (host) )))
    
    for wp in worst_processes:
        wp.start()

    for wp in worst_processes:
        wp.join()

    a2a_processes = []
    for host in host_per_switch_list:

        a2a_tm = gen_all_to_all_TM(switches, host)
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, fc_up_down_path, "A2A %d FC UP_DOWN" % (host) )))
        
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, fc_ecmp_path, "A2A %d FC ECMP" % (host) )))
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, fc_edst_path, "A2A %d FC EDST" % (host) )))
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, fc_ksp_path, "A2A %d FC KSP" % (host) )))
        
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, jf_ecmp_path, "A2A %d JF ECMP" % (host) )))
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, jf_edst_path, "A2A %d JF EDST" % (host) )))
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, jf_ksp_path, "A2A %d JF KSP" % (host) )))
        
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, xpander_ecmp_path, "A2A %d XP ECMP" % (host) )))
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, xpander_edst_path, "A2A %d XP EDST" % (host) )))
        a2a_processes.append(Process(target=Throughput, args=(a2a_tm, xpander_ksp_path, "A2A %d XP KSP" % (host) )))
    
    for ap in a2a_processes:
        ap.start()

    for ap in a2a_processes:
        ap.join()


if __name__ == "__main__":
    # test_throughput()
    worst_throughput()
