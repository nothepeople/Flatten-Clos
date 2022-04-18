from topo_repo import topology
from fc import do_topo_gen
import random
import networkx as nx



if __name__ == "__main__":
    random.seed(12)
    ports_of_vir_layer = [3,7,7,3]
    layers=len(ports_of_vir_layer)
    switches=288
    ports=24
    to_hosts=4
    res, topo_matrix, _, _ = do_topo_gen(switches, ports, to_hosts, ports_of_vir_layer)
    if res == True:
        print("FC's topology has been constructed!")
    ## construct topology object
    G = nx.Graph()

    for src in range(switches):
        for dst in range(switches):
            if topo_matrix[src][dst] == 1:
                G.add_edge(src, dst)
    # print("G's nodes: ", G.nodes)
    # print("G's edges: ", G.edges)

    print("G's nodes: ", G.number_of_nodes())
    print("G's edges: ", G.number_of_edges())
    
    tor_list = [tor_sid for tor_sid in range(switches)]
    
    
    for hosts in range(1, 21):
        demand_list = {}

        for sid in range(switches):
            demand_list[sid] = hosts

        topo = topology.Topology(G, tor_list, demand_list)
        tub = topo.get_tub()
        print("hosts = {}, TUB = {}".format(hosts, tub))
