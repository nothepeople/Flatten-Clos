import os


if __name__ == "__main__":
    """
    python2 waf --run "scratch/mesh mix/experiment/config/mesh/dcqcn/non-uniform/4pod/1.0/sr/config_0.05.txt"
    """
    CCs = ["dcqcn"]
    patterns = ["skew"] # can be uniform, incast, non-uniform
    distributions = ["web"]
    retransmits = ["pfc"] # can be PFC, IRN
    
    switches=[64]
    hot_fractions=[0.04]
    hot_traffics=[0.25, 0.50, 0.75]
    
    fat_cmd_fmt = "python2 waf --run 'scratch/fat-tree mix/experiment/config/fat-tree/%s/%s/%s/%s/%.2f/%.2f/config.txt' > out/fat-tree/%s/%s/%s/%s/%.2f/%.2f/output.txt &"
    for cc in CCs:
        for pattern in patterns:
            for sr in retransmits:
                for dis in distributions:
                    for hot_fraction in hot_fractions:
                        for hot_traffic in hot_traffics:
                            fat_cmd = fat_cmd_fmt % (cc, pattern, sr, dis, hot_fraction, hot_traffic,  cc, pattern, sr, dis, hot_fraction, hot_traffic)
                            os.system(fat_cmd)

    
    cmd_fmt = "python2 waf --run 'scratch/edst-all-in-one mix/experiment/config/%s/%s/%s/%d/%s/%s/%.2f/%.2f/config.txt' > out/%s/%s/%s/%d/%s/%s/%.2f/%.2f/output.txt &"
    for exp in ["fc-ecmp", "fc-weighted", "edst-weighted"]:
        for cc in CCs: # 1 
            for pattern in patterns: # 2
                for sw in switches:
                    for sr in retransmits: # 2
                        for dis in distributions:
                            for hot_fraction in hot_fractions:
                                for hot_traffic in hot_traffics:
                                    cmd = cmd_fmt % (exp, cc, pattern, sw, sr, dis,hot_fraction, hot_traffic,  exp, cc, pattern, sw, sr, dis, hot_fraction, hot_traffic)
                                    # print(cmd)
                                    os.system(cmd)