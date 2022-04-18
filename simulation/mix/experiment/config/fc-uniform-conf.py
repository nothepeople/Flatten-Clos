import os


contents = """ENABLE_SACK %d
ENABLE_IRN %d
ENABLE_PFC %d
ENABLE_UNIFORM_HASH 1
ENABLE_QCN 1
USE_DYNAMIC_PFC_THRESHOLD 1

PACKET_PAYLOAD_SIZE 1000

TOPOLOGY_FILE        %s
PATH_FILE             %s
FLOW_FILE            %s
TRACE_FILE           %s

PKT_OUTPUT_FILE      %s
TRACE_OUTPUT_FILE    %s
FCT_OUTPUT_FILE      %s
PFC_OUTPUT_FILE      %s
ROOT_OUTPUT_FILE     %s
VICTIM_OUTPUT_FILE   %s
TX_BYTES_OUTPUT_FILE %s

SIMULATOR_STOP_TIME %d

CC_MODE %d
ALPHA_RESUME_INTERVAL 1
RATE_DECREASE_INTERVAL 4
CLAMP_TARGET_RATE 0
RP_TIMER 900 
EWMA_GAIN 0.00390625
FAST_RECOVERY_TIMES 5
RATE_AI %dMb/s
RATE_HAI %dMb/s
MIN_RATE %dMb/s
DCTCP_RATE_AI %dMb/s

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0

HAS_WIN %d
GLOBAL_T 1
VAR_WIN 1
FAST_REACT 1
U_TARGET 0.95
MI_THRESH 0
INT_MULTI 1
MULTI_RATE 0
SAMPLE_FEEDBACK 0
PINT_LOG_BASE 1.05
PINT_PROB 1.0

RATE_BOUND 1

ACK_HIGH_PRIO 0

LINK_DOWN 0 0 0

ENABLE_TRACE 0

KMAX_MAP 4 1000000000 16 5000000000 80 10680000000 160 20000000000 320
KMIN_MAP 4 1000000000 4 5000000000 20 10680000000 40 20000000000 80
PMAX_MAP 4 1000000000 0.2 5000000000 0.2 10680000000 0.2 20000000000 0.2
PFC_MAP 4 1000000000 32 5000000000 160 10680000000 320 20000000000 400
BUFFER_SIZE 32
QLEN_MON_FILE %s
QLEN_MON_START 2030000000
QLEN_MON_END 2050000000"""

def config(topo_type, cc_type, pattern, sws, sr, distribution, client_racks, server_racks,  hot_fraction, hot_traffic, simulator_time=4):
    config_path = "fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/config.txt" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)
    # print(config_path)
    topo_file = "mix/experiment/topo/fc-uniform/%s/topo.txt" % (sws)
    path_file = "mix/experiment/topo/fc-uniform/%s/path.txt" % (sws)
    flow_file = "mix/experiment/flow/fc-uniform/%s/%s/%.2f/%.2f/flow.txt" % (pattern, distribution, hot_fraction, hot_traffic)
    trace_file = "mix/experiment/trace/fc-uniform/%s/trace.txt" % (sws)
    
    pkt_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/pkt.txt" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)
    trace_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/mix.tr" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)
    fct_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/fct.txt" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)
    pfc_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/pfc.txt" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)
    root_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/root.txt" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)
    victim_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/victim.txt" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)
    tx_bytes_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/tx_bytes.txt" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)
    qlen_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/%.2f/%.2f/qlen.txt" % (cc_type, pattern, sws, sr, distribution, hot_fraction, hot_traffic)

    if pattern=="cs-model":
        config_path = "fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/config.txt" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)
    # print(config_path)
        topo_file = "mix/experiment/topo/fc-uniform/%s/topo.txt" % (sws)
        path_file = "mix/experiment/topo/fc-uniform/%s/path.txt" % (sws)
        flow_file = "mix/experiment/flow/fc-uniform/%s/%s/C%d-S%d/flow.txt" % (pattern, distribution, client_racks, server_racks)
        trace_file = "mix/experiment/trace/fc-uniform/%s/trace.txt" % (sws)
        
        
        pkt_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/pkt.txt" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)
        trace_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/mix.tr" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)
        fct_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/fct.txt" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)
        pfc_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/pfc.txt" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)
        root_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/root.txt" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)
        victim_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/victim.txt" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)
        tx_bytes_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/tx_bytes.txt" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)
        qlen_ofile = "mix/output/fc-uniform/%s/%s/%s/%s/%s/C%d-S%d/qlen.txt" % (cc_type, pattern, sws, sr, distribution, client_racks, server_racks)

    cc_mode = 0
    if cc_type == "dcqcn":
        cc_mode = 1
    elif cc_type == "hpcc":
        cc_mode = 3
    elif cc_type == "timely":
        cc_mode = 7
    elif cc_type == "dctcp":
        cc_mode = 8
    elif cc_type == "pcn":
        cc_mode = 22
    
    
    has_win = 0
    if cc_type == "dctcp" or cc_type == "hpcc":
        has_win = 1
    
    enable_pfc = 0
    enable_sack = 0
    enable_irn = 0
    if sr == "pfc":
        enable_pfc = 1
    elif sr == "sr":
        enable_sack = 1
        has_win = 1
    elif sr == "irn":
        enable_irn = 1
        enable_sack = 1
        has_win = 1
    
    RATE_AI = 50
    RATE_HAI = 100
    MIN_RATE = 100
    DCTCP_RATE_AI = 1000
    if True:
        RATE_AI /= 10
        MIN_RATE /= 10
        RATE_HAI /= 10
        DCTCP_RATE_AI /= 10

    # print(RATE_AI, RATE_HAI, MIN_RATE, DCTCP_RATE_AI)

    config = contents % (
        enable_sack,
        enable_irn,
        enable_pfc,
        topo_file, 
        path_file,
        flow_file, 
        trace_file, 
        pkt_ofile,
        trace_ofile,
        fct_ofile,
        pfc_ofile,
        root_ofile,
        victim_ofile, 
        tx_bytes_ofile,
        simulator_time,
        cc_mode,
        RATE_AI,
        RATE_HAI,
        MIN_RATE,
        DCTCP_RATE_AI,
        has_win,
        qlen_ofile
    )
    
    config_file = open(config_path, mode='w')
    config_file.write(config)
    config_file.close()
    # print(config)

    
if __name__ == "__main__":
    patterns = ["cs-model", "skew"]
    CCs = ["dcqcn", "hpcc"]
    srs = ["pfc", "irn"]
    sws=[64]
    distributions=["web", "storage"]
    client_racks = [i * 4 for i in range(1, 9)]
    server_racks = [i * 4 for i in range(1, 9)]
    hot_fractions = [0.04, 0.10, 0.20]
    hot_traffics = [0.25, 0.50, 0.75]
    for cc in CCs:
        for pattern in patterns:
            for sw in sws:
                for sr in srs:
                    for dis in distributions:
                        for client in client_racks:
                            for server in server_racks:
                                for hot_fraction in hot_fractions:
                                    for hot_traffic in hot_traffics:
                                        config("fc-uniform", cc, pattern, sw, sr, dis, client, server, hot_fraction, hot_traffic)
