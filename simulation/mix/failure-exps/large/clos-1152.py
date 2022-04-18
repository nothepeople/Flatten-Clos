# !/usr/bin/python3
# coding=utf-8
from optparse import OptionParser

def fat_tree(host_bandwidth=1, topo_dir="", trace_dir=""):
    pod_num = 8
    host_per_ToR = 12
    tor_switches_num = 12 * 8
    agg_switches_num = 12 * 8
    core_switches_num = 12 * 12
    hosts_num = host_per_ToR * tor_switches_num
    # print("%d pods" % (pod_num))
    print(f"ToR: {tor_switches_num}, Agg: {agg_switches_num}, Core: {core_switches_num} Pods: {pod_num}, host: {hosts_num}" )
    host_ids = [id for id in range(hosts_num)]
    tor_switch_ids = [id + hosts_num for id in range(tor_switches_num)]
    agg_switches_ids = [hosts_num + tor_switches_num + id for id in range(agg_switches_num)]
    core_switch_ids = [id + hosts_num + tor_switches_num + agg_switches_num for id in range(core_switches_num)]

    """
    calculate bandwidth
    """
    h2tor_link = hosts_num
    tor2agg_link = 12 * 12 * 8
    agg2core_link = 12 * 12 * 8
    print("host links: %d, Tor_to_Agg links: %d Agg_to_Core links: %d" % (h2tor_link, tor2agg_link, agg2core_link))

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
        topo_file.write("%d %d %.2fGbps %dns 0.000000\n" % (host_id, tor_switch_ids[host_id // host_per_ToR], 1, 1000) )
        
    ## connect ToR switches to Agg switches
    tor_per_pod = tor_switches_num // pod_num
    agg_per_pod = agg_switches_num // pod_num
    print("ToR_Per_Pod: {} Agg_Per_Pod: {}".format(tor_per_pod, agg_per_pod))
    for i in range(tor_switches_num):
        pod = i // tor_per_pod
        for j in range(agg_per_pod):
            tor_idx = i
            agg_idx = pod * agg_per_pod + j
            topo_file.write("%d %d %.2fGbps %dns 0.000000\n" % (tor_switch_ids[tor_idx], agg_switches_ids[agg_idx], 1, 1000) )

    ## connect Agg switches to Core switches
    for i in range(pod_num):
        for j in range(agg_per_pod):
            agg_idx = i * agg_per_pod +  j
            ## connect agg to core switch 
            for k in range(agg_per_pod):
                core_idx = j * agg_per_pod + k
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
    # parser = OptionParser()
    # parser.add_option("-b", "--bandwidth", dest = "bandwidth", help = "the bandwidth of host link, y default 1G", default = "1")
    # options, args = parser.parse_args()
    # link_bw = int(options.bandwidth)

    topo_fmt = "./topo/clos/topo.txt"
    trace_fmt = "./trace/clos/trace.txt"
    
    fat_tree(
      host_bandwidth=1,
      topo_dir=(topo_fmt), 
      trace_dir=(trace_fmt)
    )