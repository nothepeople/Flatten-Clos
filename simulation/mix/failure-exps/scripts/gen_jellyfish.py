import numpy as np
import random
import time



def is_sw_having_remain_port(sw, to_hosts, port_occupy_map):
    return False in port_occupy_map[sw][to_hosts:]

def get_sw_num_unoccupied_ports(sw, to_hosts, port_occupy_map):
    num = 0
    for unoccupied in port_occupy_map[sw][to_hosts:]:
        if unoccupied == False: num+=1
    return num

def get_sw_unoccupied_port(sw, to_hosts, port_occupy_map):
    ports = port_occupy_map[sw][to_hosts:]
    rnd = random.randint(0, len(ports)-1)
    selected_port_no = to_hosts +rnd
    while port_occupy_map[sw][selected_port_no]:
        rnd = random.randint(0, len(ports)-1)
        selected_port_no = to_hosts +rnd
    return selected_port_no

def wiring(sA, sB, to_hosts, port_occupy_map):
    pA = get_sw_unoccupied_port(sA, to_hosts, port_occupy_map)
    pB = get_sw_unoccupied_port(sB, to_hosts, port_occupy_map)
    if pA != -1 and pB != -1:
        port_occupy_map[sA][pA] = True
        port_occupy_map[sB][pB] = True
        return True, pA, pB
    return False, -1, -1

def can_be_over(switches, port_occupy_map, to_hosts):
    for sid in range(switches):
       if False in port_occupy_map[sid][to_hosts:]:
           return False
    return True

def do_jellyfish_topo_gen(switches, ports, to_hosts):
    links = []
    port_occupy_map = {}
    for sid in range(switches):
        port_occupy_map[sid] = [False for i in range(ports)]
    # print("port occupy map {}".format(port_occupy_map))
    switch_ids = [id for id in range(switches)]
    topo_matrix = np.zeros(shape=(switches, switches), dtype=int)
    ports_conn_matrix = {}
    for i in range(switches):
        ports_conn_matrix[i] = {}
        for j in range(switches):
            ports_conn_matrix[i][j] = []
    
    consecFails = 0
    while len(switch_ids) > 1 and consecFails < 10000:
        sA =  random.choice(switch_ids)
        sB =  random.choice(switch_ids)
        # print("check")
        if sA == sB:
            continue
        
        if topo_matrix[sA][sB] == 1 and topo_matrix[sB][sA] == 1: 
            consecFails += 1
            continue 
        # print("check over")
        res, pA, pB = wiring(sA, sB, to_hosts, port_occupy_map)
        # time.sleep(1)
        # print("{} {} {} {} {} len sids{}".format(res, sA, sB, pA, pB, len(switch_ids)))
        if res == True:
            consecFails = 0

            links.append([sA, sB])
            links.append([sB, sA])

            topo_matrix[sA][sB] = 1
            topo_matrix[sB][sA] = 1
            ports_conn_matrix[sA][sB].extend([pA, pB])
            ports_conn_matrix[sB][sA].extend([pB, pA])
            if not is_sw_having_remain_port(sA, to_hosts, port_occupy_map): switch_ids.remove(sA)
            if not is_sw_having_remain_port(sB, to_hosts, port_occupy_map): switch_ids.remove(sB)
            if can_be_over(switches, port_occupy_map, to_hosts):
                break
    # print("break remain switches {}".format(len(switch_ids)))
    if len(switch_ids) > 0:
        return False, None, None
    ## Readjust Links
    return True, topo_matrix, ports_conn_matrix

def jellyfish_topo_gen(switches, ports, to_hosts, mat_dir=None):
    ret, topo_matrix, ports_conn_matrix = do_jellyfish_topo_gen(switches, ports, to_hosts)
    while not ret:
        # print ( "retry" )
        ret, topo_matrix, ports_conn_matrix = do_jellyfish_topo_gen(switches, ports, to_hosts)
    # print("success")
    if mat_dir is not None:
        print("Write Jellyfish mat")
        mat_file = open(mat_dir, mode='w')
        mat_file.write("%d %d %d %d\n" % (to_hosts, switches, (ports - to_hosts) * switches, switches * to_hosts) )
        for i in range(switches):
            for j in range(switches):
                if i==j or topo_matrix[i,j]==0: continue
                mat_file.write("%d %d %d %d\n"%(i, j, ports_conn_matrix[i][j][0], ports_conn_matrix[i][j][1]))
        mat_file.close()
    return topo_matrix, ports_conn_matrix



if __name__ == "__main__":
   for i in range(100):
        random.seed(i)
        jellyfish_topo_gen(64, 16, 4)
