# coding=utf-8
from optparse import OptionParser

def fat_tree(pod_num=4, hosts_per_tor=8, host_bandwidth=1, topo_dir="", trace_dir="", over_subscription="1:1:1"):
    tor_switches_num = 32
    agg_switches_num = 32
    core_switches_num = 16
    hosts_num = 256
    # print("%d pods" % (pod_num))
    # print("In each pod, tor switches: %d, agg switches: %d" % (tor_per_pod, agg_per_pod))
    host_ids = [id for id in range(hosts_num)]
    tor_switch_ids = [id + hosts_num for id in range(tor_switches_num)]
    agg_switches_ids = [hosts_num + tor_switches_num + id for id in range(agg_switches_num)]
    core_switch_ids = [id + hosts_num + tor_switches_num + agg_switches_num for id in range(core_switches_num)]
    # print("tor: ", tor_switch_ids)
    # print("agg: ", agg_switches_ids)
    # print("core: ", core_switch_ids)
    """
    calculate bandwidth
    """
    h2tor_link = hosts_num
    tor2agg_link = 256
    agg2core_link = 256
    # print("host links: %d, tor to core links: %d" % (h2tor_link, tor2core_link))

    # h2tor_bw = host_bandwidth
    # tor2core_bw = float(h2tor_link * host_bandwidth) / tor2core_link
    
    # print("bandwidth info...")
    # print("host bw: %.2fGbps, tor to core bw: %.2fGbps" % (h2tor_bw, tor2core_bw))

    
    # topo is the topo file path
    topo_file = open(topo_dir, mode='w')
    links = h2tor_link + tor2agg_link + agg2core_link
    total_nodes = hosts_num + tor_switches_num + core_switches_num + agg_switches_num
    switch_nodes = tor_switches_num + core_switches_num + agg_switches_num
    topo_file.write("%d %d %d\n" % (total_nodes, switch_nodes, links) )

    switches = []
    switches.extend(tor_switch_ids)
    switches.extend(agg_switches_ids)
    switches.extend(core_switch_ids)
    # print("switch ids: ", switches)
    topo_file.write("%s\n" % ( " ".join(map(str, switches)) ) )
    
    ## connect host to ToR switches.
    for host_id in host_ids:
        # print('hid', host_id)
        topo_file.write("%d %d %.2fGbps %dns 0.000000\n" % (host_id, tor_switch_ids[host_id / hosts_per_tor], 1, 1000) )
        
    ## connect ToR switches to Agg switches
    tor_per_pod = tor_switches_num / pod_num
    agg_per_pod = agg_switches_num / pod_num
    for i in range(tor_switches_num):
        pod = i / tor_per_pod
        for j in range(agg_per_pod):
            tor_idx = i
            agg_idx = pod * agg_per_pod + j
            topo_file.write("%d %d %.2fGbps %dns 0.000000\n" % (tor_switch_ids[tor_idx], agg_switches_ids[agg_idx], 1, 1000) )

    ## connect Agg switches to Core switches
    # print("agg idxs: ", agg_switches_ids)
    core_per_pod = core_switches_num / pod_num
    for pod in range(4):
        for i in range(core_per_pod):
            core_idx = pod * core_per_pod + i
            if i < 2:
                for j in range(pod_num):
                    for k in range(4):
                        agg_idx = j * agg_per_pod + k
                        topo_file.write("%d %d %.2fGbps %dns 0.000000\n" % (agg_switches_ids[agg_idx], core_switch_ids[core_idx], 1, 1000) )
            else:
                for j in range(pod_num):
                    for k in range(4, 8):
                        agg_idx = j * agg_per_pod + k
                        topo_file.write("%d %d %.2fGbps %dns 0.000000\n" % (agg_switches_ids[agg_idx], core_switch_ids[core_idx], 1, 1000) )
    topo_file.close()

    """
    generate trace file
    """
    trace_file = open(trace_dir, mode='w')
    trace_file.write("%d\n" % (total_nodes))
    trace_file.write("%s\n" % (" ".join( map(str, [id for id in range(total_nodes)]) ) ) )
    trace_file.close()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--bandwidth", dest = "bandwidth", help = "the bandwidth of host link, y default 1G", default = "1")
    options, args = parser.parse_args()
    link_bw = int(options.bandwidth)

    topo_fmt = "../experiment/topo/fat-tree/topo.txt"
    trace_fmt = "../experiment/trace/fat-tree/trace.txt"
    
    fat_tree(
      pod_num=4,
      hosts_per_tor=8,
      host_bandwidth=1,
      topo_dir=(topo_fmt), 
      trace_dir=(trace_fmt)
    )
    # pods = [4, 8, 16]
    # bws = ["1G", "40G"]
    # for pod in pods:
    #     for bw in bws:
    #         host_bw = int(bw.split('G')[0])
    #         # print(bw, host_bw, pod)
    #         fat_tree(
    #             pod_num=pod,
    #             hosts_per_tor=pod,
    #             host_bandwidth=host_bw,
    #             topo_dir=(topo_fmt % (pod, bw)), 
    #             trace_dir=(trace_fmt % (pod))
    #             )
