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
from gen_jellyfish import jellyfish_topo_gen
from gen_xpander import Xpander_topo_gen

def edst_path_calculation(mat_dir, path_dir, k):
    os.system("./edst %s %s %d" % (mat_dir, path_dir, k))

def path_analysis():
    
    ports_of_vir_layer1 = [4,8,8,4]
    switches1=512
    ports1=24
    to_hosts1=0

    ports_of_vir_layer2 = [2, 6, 6, 2]
    switches2=128
    ports2=16
    to_hosts2=0

    ports_of_vir_layer3 = [3, 7, 7, 3]
    switches3=285
    ports3=20
    to_hosts3=0
    
    """
    host_per_switch_list = [4]
    # xrandom.seed(1)
    ports_of_vir_layer = [2,4,4,2]
    layers=len(ports_of_vir_layer)
    switches=65
    ports=16
    to_hosts=4
    """

    fc_mat_dir1 = "throughput/fc1_mat.txt"
    fc_mat_dir2 = "throughput/fc2_mat.txt"
    fc_mat_dir3 = "throughput/fc3_mat.txt"
    
    fc_path_dir1 = "throughput/fc_edst_path1.txt"
    fc_path_dir2 = "throughput/fc_edst_path2.txt"
    fc_path_dir3 = "throughput/fc_edst_path3.txt"
    
    print("FC-1 UP-DOWN PATH CALCULATION...")
    fc_topo_matrix1, ports_conn_matrix1, switch_objs1 = fc_topo_gen(switches1, ports1, to_hosts1, ports_of_vir_layer1, fc_mat_dir1)
    vclos1 = convert_into_vclos(switch_objs1, fc_topo_matrix1, ports_conn_matrix1, ports_of_vir_layer1)
    _ = paths_of_any_pairs(vclos1, len(ports_of_vir_layer1), switches1)
    
    exit(0)
    print("FC-2 UP-DOWN PATH CALCULATION...")
    fc_topo_matrix2, ports_conn_matrix2, switch_objs2 = fc_topo_gen(switches2, ports2, to_hosts2, ports_of_vir_layer2, fc_mat_dir2)
    vclos2 = convert_into_vclos(switch_objs2, fc_topo_matrix2, ports_conn_matrix2, ports_of_vir_layer2)
    _ = paths_of_any_pairs(vclos2, len(ports_of_vir_layer2), switches2)

    print("FC-3 UP-DOWN PATH CALCULATION...")
    fc_topo_matrix3, ports_conn_matrix3, switch_objs3 = fc_topo_gen(switches3, ports3, to_hosts3, ports_of_vir_layer3, fc_mat_dir3)
    vclos3 = convert_into_vclos(switch_objs3, fc_topo_matrix3, ports_conn_matrix3, ports_of_vir_layer3)
    _ = paths_of_any_pairs(vclos3, len(ports_of_vir_layer3), switches3)
    
    _ = cal_shortest_path(fc_topo_matrix1, switches1)
    _ = cal_shortest_path(fc_topo_matrix2, switches2)
    _ = cal_shortest_path(fc_topo_matrix3, switches3) 

    print("CALCULATE EDST AND KSP PATHS...")
    os.system("g++ edst.cc -w -std=c++11 -o edst")

    path_processes = []
    path_processes.append(Process(target=edst_path_calculation, args=(fc_mat_dir1, fc_path_dir1, (ports1 - to_hosts1) / 2 )))
    path_processes.append(Process(target=edst_path_calculation, args=(fc_mat_dir2, fc_path_dir2, (ports2 - to_hosts2) / 2 )))
    path_processes.append(Process(target=edst_path_calculation, args=(fc_mat_dir3, fc_path_dir3, (ports3 - to_hosts3) / 2 )))
    
    for p in path_processes:
        p.start()

    for p in path_processes:
        p.join()


if __name__ == "__main__":
    path_analysis()
