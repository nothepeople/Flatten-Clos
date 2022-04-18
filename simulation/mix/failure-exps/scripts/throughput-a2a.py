#!/usr/bin/env python3
import imp
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
from throughput import Throughput

from gen_jellyfish import jellyfish_topo_gen
from gen_xpander import Xpander_topo_gen


import sys
sys.path.append('TUB')
from topo_repo import topology

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

def path_calculation(type, mat_dir, path_dir):
    if type == "ksp": os.system("./ksp %s %s" % (mat_dir, path_dir))
    else: os.system("./edst %s %s" % (mat_dir, path_dir))

def a2a_throughput():
    # host_per_switch_list = [6, 8, 12, 18]
    # random.seed(1)
    # ports_of_vir_layer = [3,7,7,3]
    # layers=len(ports_of_vir_layer)
    # switches=285
    # ports=24
    # to_hosts=4

    host_per_switch_list = [4]
    random.seed(1)
    ports_of_vir_layer = [2,4,4,2]
    layers=len(ports_of_vir_layer)
    switches=65
    ports=16
    to_hosts=4

    fc_mat_dir = "throughput/a2a/fc/mat.txt"
    jf_mat_dir = "throughput/a2a/jf/mat.txt"
    xpander_mat_dir = "throughput/a2a/xpander/mat.txt"

    fc_ksp_path_dir = "throughput/a2a/fc/ksp_path.txt"
    jf_ksp_path_dir = "throughput/a2a/jf/ksp_path.txt"
    xpander_ksp_path_dir =  "throughput/a2a/xpander/ksp_path.txt"

    fc_edst_path_dir = "throughput/a2a/fc/edst_path.txt"
    jf_edst_path_dir = "throughput/a2a/jf/edst_path.txt"
    xpander_edst_path_dir =  "throughput/a2a/xpander/edst_path.txt"

    jf_topo_matrix, _ = jellyfish_topo_gen(switches, ports, to_hosts, jf_mat_dir)
    xpander_topo_matrix, _ = Xpander_topo_gen(switches, ports, to_hosts, xpander_mat_dir)

    fc_topo_matrix, ports_conn_matrix, switch_objs = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer, fc_mat_dir)

    vclos = convert_into_vclos(switch_objs, fc_topo_matrix, ports_conn_matrix, ports_of_vir_layer)
    fc_up_down_path = paths_of_any_pairs(vclos, len(ports_of_vir_layer), switches)
    
    fc_ecmp_path = cal_shortest_path(fc_topo_matrix, switches)
    jf_ecmp_path = cal_shortest_path(jf_topo_matrix, switches)
    xpander_ecmp_path = cal_shortest_path(xpander_topo_matrix, switches) 

    print("CALCULATE EDST AND KSP PATHS...")
    os.system("g++ edst.cc -w -std=c++11 -o edst")
    os.system("g++ k-short-path.cc -w -std=c++11 -o ksp")

    
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
    throughput_processes = []
    for host in host_per_switch_list:
        a2a_tm = gen_all_to_all_TM(switches, host)
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, fc_up_down_path, "%d FC UP_DOWN" % (host) )))
        
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, fc_ecmp_path, "%d FC ECMP" % (host) )))
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, fc_edst_path, "%d FC EDST" % (host) )))
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, fc_ksp_path, "%d FC KSP" % (host) )))
        
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, jf_ecmp_path, "%d JF ECMP" % (host) )))
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, jf_edst_path, "%d JF EDST" % (host) )))
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, jf_ksp_path, "%d JF KSP" % (host) )))
        
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, xpander_ecmp_path, "%d XP ECMP" % (host) )))
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, xpander_edst_path, "%d XP EDST" % (host) )))
        throughput_processes.append(Process(target=Throughput, args=(a2a_tm, xpander_ksp_path, "%d XP KSP" % (host) )))
    
    for p in throughput_processes:
        p.start()

    for p in throughput_processes:
        p.join()






if __name__ == "__main__":
    # test_throughput()
    a2a_throughput()
