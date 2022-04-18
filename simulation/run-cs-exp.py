import os

# python2 waf --run "scratch/edst-all-in-one mix/experiment/config/fc-lp/dcqcn/cs-model/64/pfc/web/C4-S4/config.txt"
if __name__ == "__main__":
    """
    python2 waf --run "scratch/mesh mix/experiment/config/mesh/dcqcn/non-uniform/4pod/1.0/sr/config_0.05.txt"
    """
    CCs = ["dcqcn"]
    patterns = ["cs-model"] # can be uniform, incast, non-uniform
    distributions = ["web"]
    retransmits = ["pfc"] # can be PFC, IRN
    
    switches=[64]

    client_racks = [i * 4 for i in range(1, 9)]
    server_racks = [i * 4 for i in range(1, 9)]

    """
    fat_cmd_fmt = "python2 waf --run 'scratch/fat-tree mix/experiment/config/fat-tree/%s/%s/%s/%s/C%d-S%d/config.txt' > out/fat-tree/%s/%s/%s/%s/C%d-S%d/output.txt &"
    for cc in CCs:
        for pattern in patterns:
            for sr in retransmits:
                for dis in distributions:
                    for client in client_racks:
                        for server in server_racks:
                            fat_cmd = fat_cmd_fmt % (cc, pattern, sr, dis, client, server,  cc, pattern, sr, dis, client, server)
                            os.system(fat_cmd)
    """

    """
    DONE: clos, fc-ecmp, fc-weighted, fc-lp, edst-weighted, edst-lp, edst-uniform
    """

    exps1 = ["fc-ecmp", "fc", "edst", "fc-weighted", "edst-weighted", "fc-uniform", "edst-uniform"] 

    cmd_fmt = "python2 waf --run 'scratch/edst-all-in-one mix/experiment/config/%s/%s/%s/%d/%s/%s/C%d-S%d/config.txt' > out/%s/%s/%s/%d/%s/%s/C%d-S%d/output.txt &"
    for exp in ["fc-uniform"]:
        for cc in CCs: # 1 
            for pattern in patterns: # 2
                for sw in switches:
                    for sr in ["pfc"]: # 2
                        for dis in distributions:
                            for client in client_racks:
                                for server in server_racks:
                                    cmd = cmd_fmt % (exp, cc, pattern, sw, sr, dis, client, server,   exp, cc, pattern, sw, sr, dis, client, server)
                                    os.system(cmd)


