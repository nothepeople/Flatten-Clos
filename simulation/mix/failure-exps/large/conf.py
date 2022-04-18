import os


contents = """ENABLE_SACK 0
ENABLE_IRN 0
ENABLE_PFC 1
%s
ENABLE_QCN 1
USE_DYNAMIC_PFC_THRESHOLD 1

PACKET_PAYLOAD_SIZE 1000

TOPOLOGY_FILE        %s
%s
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

CC_MODE 1
ALPHA_RESUME_INTERVAL 1
RATE_DECREASE_INTERVAL 4
CLAMP_TARGET_RATE 0
RP_TIMER 900 
EWMA_GAIN 0.00390625
FAST_RECOVERY_TIMES 5
RATE_AI 5Mb/s
RATE_HAI 10Mb/s
MIN_RATE 10Mb/s
DCTCP_RATE_AI 100Mb/s

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0

HAS_WIN 0
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

def config(routing, pattern, rate, simulator_time=4):
    config_path = "config/%s/%s/%s/config.txt" % (routing, pattern, rate)
    # print(config_path)
    topo_file = "mix/failure-exps/large/topo/%s/topo.txt" % (routing)
    # path
    path_file = ""
    enable_weighted_hash=""
    if routing == "up-down":
        enable_weighted_hash = "ENABLE_WEIGHTED_HASH 1"
        path_file = "PATH_FILE            mix/failure-exps/large/topo/up-down/path.txt"

    if routing == "edst":
        enable_weighted_hash = "ENABLE_WEIGHTED_HASH 1"
        path_file = "PATH_FILE            mix/failure-exps/large/topo/edst/path.txt"

    flow_file = "mix/failure-exps/large/traffic/%s/%s/flow.txt" % (pattern, rate)
    trace_file = "mix/failure-exps/large/trace/%s/trace.txt" % (routing)
    
    pkt_ofile = "mix/failure-exps/large/output/%s/%s/%s/pkt.txt" % (routing, pattern, rate)
    trace_ofile = "mix/failure-exps/large/output/%s/%s/%s/mix.tr" % (routing, pattern, rate)
    fct_ofile = "mix/failure-exps/large/output/%s/%s/%s/fct.txt" % (routing, pattern, rate)
    pfc_ofile = "mix/failure-exps/large/output/%s/%s/%s/pfc.txt" % (routing, pattern, rate)
    root_ofile = "mix/failure-exps/large/output/%s/%s/%s/root.txt" % (routing, pattern, rate)
    victim_ofile = "mix/failure-exps/large/output/%s/%s/%s/victim.txt" % (routing, pattern, rate)
    tx_bytes_ofile = "mix/failure-exps/large/output/%s/%s/%s/tx_bytes.txt" % (routing, pattern, rate)
    qlen_ofile = "mix/failure-exps/large/output/%s/%s/%s/qlen.txt" % (routing, pattern, rate)

    config = contents % (
        enable_weighted_hash,
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
        qlen_ofile
    )
    
    config_file = open(config_path, mode='w')
    config_file.write(config)
    config_file.close()
    
if __name__ == "__main__": 
    routings = ["clos", "ecmp", "up-down", "edst"]
    patterns = ["a2a", "o2o", "skewed", "elephant"]
    rates = [30, 70, 100]
    for routing in routings:
        for pattern in patterns:
            for rate in rates:
                config(routing, pattern, rate)
