from fc import fc_topo_gen
from gen_jellyfish import jellyfish_topo_gen
from gen_xpander import Xpander_topo_gen
from ecmp import cal_shortest_path
from deadlock_detection import deadlock_detection
import random, time
from multiprocessing import Process, Value, Array

def choose_cs_racks(switches, client_racks, server_racks):
    # print("c-rack {} s-rack {}".format(client_racks, server_racks) )
    client_rack_ids = []
    server_rack_ids = []
    for _ in range(client_racks):
        client_id = random.randint(0, switches-1)
        while client_id in client_rack_ids:
            client_id = random.randint(0, switches-1)
        client_rack_ids.append(client_id)
        # print()
    
    for _ in range(server_racks):
        server_id = random.randint(0, switches-1) 
        while server_id in client_rack_ids or server_id in server_rack_ids:
            server_id = random.randint(0, switches-1)
        server_rack_ids.append(server_id)
    # print(client_rack_ids, server_rack_ids)
    return client_rack_ids, server_rack_ids

def cs_analysis(switches, ports, topo_matrix, ports_conn_matrix, inters=4):
    
    client_racks = [4 * i for i in range(1,switches / (2 * inters) )]
    server_racks = [4 * i for i in range(1, switches / (2 * inters) )]
    
    shortest_path_pair = cal_shortest_path(topo_matrix, switches)

    # print(client_racks, server_racks)
    ## 
    deadlock = 0
    prob = 0
    for clients in client_racks:
        for servers in server_racks:
            client_rack_ids, server_rack_ids = choose_cs_racks(switches, clients, servers)
            
            paths = []
            for client_id in client_rack_ids:
                for server_id in server_rack_ids:
                    # print("cs" , shortest_path_pair[client_id][server_id] )
                    paths.extend( shortest_path_pair[client_id][server_id] )
            # print("paths {}".format(paths))
            has_cycle, _, _, _ = deadlock_detection(paths, switches, ports, ports_conn_matrix)
            if has_cycle:
                deadlock += 1
    # print(" Deadlock Prob in C/S Traffic Evaluation: %.2f" % (deadlock / float(64)))
    exps = len(client_racks) * len(client_racks)
    return deadlock / float(exps)

def skewed_analysis():
    pass

def fc_cs_test(test_times):
    print("fc process start@{}".format( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ) )
    bg_time = time.time()
    fc_config=[
        (32, 10, [3,5,2]),
        (64, 12, [2,4,4,2]),
        (128, 16, [3,5,5,3])
    ]
    for cfg in fc_config:
        prob = 0
        for i in range(test_times):
            # print("{}th".format(i))
            topo_matrix, ports_conn_matrix = fc_topo_gen(cfg[0], cfg[1], 0, cfg[2])
            prob += cs_analysis(cfg[0], cfg[1], topo_matrix, ports_conn_matrix)
        print("Cfg {} after {} tests, deadlock prob of C/S evaluation in FC is: {:.2f}".format(cfg, test_times, prob / test_times))
    
    ed_time = time.time()
    print("xpander process run {}s".format(ed_time - bg_time))


def xpander_cs_test(test_times):
    print("xpander process start@{}".format( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ) )
    bg_time = time.time()
    xpander_config = [
        (32, 7), 
        (64, 15), 
        (128, 15)]
    for cfg in xpander_config:
        prob = 0
        for i in range(test_times):
            # print("{}th".format(i))
            topo_matrix, ports_conn_matrix = Xpander_topo_gen(cfg[0], cfg[1])
            prob += cs_analysis(cfg[0], cfg[1], topo_matrix, ports_conn_matrix)
        print("Cfg {} after {} tests, deadlock prob of C/S evaluation in Xpander is: {:.2f}".format(cfg, test_times, prob / test_times))

    ed_time = time.time()
    print("xpander process run {}s".format(ed_time - bg_time))

def jellyfish_cs_test(test_times):
    print("jellyfish process start@{}".format( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ) )
    bg_time = time.time()
    jf_config = [
        (32, 7), 
        (64, 12), 
        (128, 16)]
    for cfg in jf_config:
        prob = 0
        for i in range(test_times):
            # print("{}th".format(i))
            topo_matrix, ports_conn_matrix = jellyfish_topo_gen(cfg[0], cfg[1], 0)
            # print("je")
            prob += cs_analysis(cfg[0], cfg[1], topo_matrix, ports_conn_matrix)
        print("Cfg: {}, after {} tests, deadlock prob of C/S evaluation in Jellyfish is: {:.2f}".format(cfg, test_times, prob / test_times))
    ed_time = time.time()
    print("xpander process run {}s".format(ed_time - bg_time))


def cs_test():
    test_times = 10
    fc_process = Process(target=fc_cs_test, args=(test_times,))
    xpander_process = Process(target=xpander_cs_test, args=(test_times,))
    jf_process = Process(target=jellyfish_cs_test, args=(test_times,))

    fc_process.start()
    xpander_process.start()
    jf_process.start()

    fc_process.join()
    xpander_process.join()
    jf_process.join()

    print("over")
    ## (switches, ports)

if __name__ == "__main__":
    cs_test()