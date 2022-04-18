from gen_jellyfish import jellyfish_topo_gen
from gen_xpander import Xpander_topo_gen
from fc import fc_topo_gen
from ecmp import cal_shortest_path
from deadlock_detection import deadlock_detection
import random
import time

def get_shortest_path_list(topo_matrix, switches):
    sp_list = []
    shortest_path_pair = cal_shortest_path(topo_matrix, switches)
    for src in range(switches):
        for dst in range(switches):
            if src == dst: continue
            for path in shortest_path_pair[src][dst]:
                sp_list.append(path)
    return sp_list


def jellyfish_analysis(switches, port_count, to_hosts=0, test_times=10):
    deadlock_nums = 0
    for i in range(test_times):
        random.seed(time.time())
        topo_matrix, ports_conn_matrix = jellyfish_topo_gen(switches, port_count, to_hosts)
        print("jen topo over")
        sp_list = get_shortest_path_list(topo_matrix, switches)
        # print(sp_list)
        has_cycle, _, _, _ = deadlock_detection(sp_list, switches, port_count, ports_conn_matrix)
        if has_cycle:
            deadlock_nums += 1
        print(i)
    print("When apply ECMP in jellfish has deadlock prob is: %.2f" % (deadlock_nums / float(test_times)))

def xpander_analysis(switches, port_degree, test_times=10):
    deadlock_nums = 0
    for i in range(test_times):
        random.seed(time.time())
        topo_matrix, ports_conn_matrix = Xpander_topo_gen(switches, port_degree)
        # print("xpander topo over")
        sp_list = get_shortest_path_list(topo_matrix, switches)
        # print(sp_list)
        has_cycle, _, _, _ = deadlock_detection(sp_list, switches, port_degree, ports_conn_matrix)
        if has_cycle:
            deadlock_nums += 1
        # print(i)
    print("When apply ECMP in Xpander has deadlock prob is: %.2f" % (deadlock_nums / float(test_times)))


def fc_analysis(switches, ports, to_hosts, ports_of_vir_layer, test_times=10):
    deadlock_nums = 0
    for i in range(test_times):
        random.seed(time.time())
        topo_matrix, ports_conn_matrix = fc_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
        # print("FC topo over")
        sp_list = get_shortest_path_list(topo_matrix, switches)
        # print(sp_list)
        has_cycle, _, _, _ = deadlock_detection(sp_list, switches, ports, ports_conn_matrix)
        if has_cycle:
            deadlock_nums += 1
        # print(i)
    print("When apply ECMP in Flattened Clos has deadlock prob is: %.2f" % (deadlock_nums / float(test_times)))

if __name__=="__main__":
    port_count = [8, 12, 16]
    switches = [16, 64, 256]
    tests = 100
    count = 0
    jellyfish_analysis(3, 2)
    # xpander_analysis(64, 15)
    # fc_analysis(switches=64, ports=12, to_hosts=0, ports_of_vir_layer=[2,4,4,2])

